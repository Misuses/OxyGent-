# final_long_running_benchmark.py
import asyncio
import time
import aiohttp
import json
import csv
import psutil
import subprocess
import pandas as pd
from datetime import datetime, timedelta
import argparse
import os
import numpy as np
from collections import deque

class FinalLongRunningBenchmark:
    def __init__(self, args):
        self.args = args
        self.performance_data = []
        self.batch_performance_data = []  # 每批请求的性能数据
        self.start_time = None
        self.end_time = None
        self.request_counter = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        
        # 批处理统计
        self.batch_size = args.batch_size
        self.current_batch_requests = 0
        self.current_batch_tokens = 0
        self.current_batch_start_time = None
        self.last_batch_time = None
        
        # 滑动窗口统计
        self.recent_rps = deque(maxlen=10)  # 最近10批的RPS
        self.recent_tps = deque(maxlen=10)  # 最近10批的TPS
        
        # 历史数据存储（用于生成报告）
        self.hourly_performance = []
        
    async def get_gpu_metrics(self):
        """获取GPU指标"""
        try:
            result = subprocess.check_output([
                'nvidia-smi', 
                '--query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu,utilization.memory',
                '--format=csv,noheader,nounits'
            ], encoding='utf-8', timeout=10)
            
            lines = result.strip().split('\n')
            gpu_data = []
            
            for line in lines:
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 5:
                    def safe_float(value):
                        try:
                            return float(value) if value and value != '[N/A]' else 0.0
                        except:
                            return 0.0
                    
                    gpu_data.append({
                        'timestamp': parts[0],
                        'gpu_temp': safe_float(parts[1]),
                        'power_draw': safe_float(parts[2]),
                        'gpu_util': safe_float(parts[3]),
                        'memory_util': safe_float(parts[4])
                    })
            
            return gpu_data
        except Exception as e:
            print(f"获取GPU指标失败: {e}")
            return [{
                'timestamp': datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                'gpu_temp': 0,
                'power_draw': 0,
                'gpu_util': 0,
                'memory_util': 0
            }]
    
    async def get_system_metrics(self):
        """获取系统指标"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        
        return {
            'timestamp': datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'disk_read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
            'disk_write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0
        }
    
    async def record_batch_performance(self):
        """记录一批请求的性能指标"""
        if self.current_batch_requests == 0:
            return
            
        current_time = time.time()
        batch_duration = current_time - self.current_batch_start_time
        
        # 计算这批请求的性能
        batch_rps = self.current_batch_requests / batch_duration if batch_duration > 0 else 0
        batch_tps = self.current_batch_tokens / batch_duration if batch_duration > 0 else 0
        
        # 获取系统指标
        gpu_metrics = await self.get_gpu_metrics()
        system_metrics = await self.get_system_metrics()
        
        batch_data = {
            'batch_timestamp': datetime.now().isoformat(),
            'elapsed_minutes': (current_time - self.start_time) / 60,
            'elapsed_hours': (current_time - self.start_time) / 3600,
            'batch_duration_seconds': batch_duration,
            'batch_requests': self.current_batch_requests,
            'batch_tokens': self.current_batch_tokens,
            'batch_rps': batch_rps,
            'batch_tps': batch_tps,
            'cumulative_requests': self.request_counter,
            'cumulative_tokens': self.total_tokens,
            'success_rate': self.successful_requests / max(self.request_counter, 1),
            **system_metrics
        }
        
        # 添加GPU指标
        if gpu_metrics:
            batch_data.update(gpu_metrics[0])
        
        self.batch_performance_data.append(batch_data)
        self.recent_rps.append(batch_rps)
        self.recent_tps.append(batch_tps)
        
        # 输出实时性能
        avg_rps = np.mean(self.recent_rps) if self.recent_rps else 0
        avg_tps = np.mean(self.recent_tps) if self.recent_tps else 0
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"批次: {len(self.batch_performance_data):4d} | "
              f"瞬时RPS: {batch_rps:6.2f} | "
              f"瞬时TPS: {batch_tps:6.2f} | "
              f"平均RPS: {avg_rps:6.2f} | "
              f"平均TPS: {avg_tps:6.2f} | "
              f"GPU: {batch_data.get('gpu_temp', 0):3.0f}°C/{batch_data.get('gpu_util', 0):3.0f}% | "
              f"功耗: {batch_data.get('power_draw', 0):5.1f}W | "
              f"成功率: {batch_data['success_rate']:5.1%}")
        
        # 保存到CSV
        self.save_batch_metrics_to_csv()
        
        # 重置当前批次统计
        self.current_batch_requests = 0
        self.current_batch_tokens = 0
        self.current_batch_start_time = current_time
        self.last_batch_time = current_time
    
    async def send_request(self, session, prompt, request_id):
        """发送单个请求"""
        payload = {
            "prompt": prompt,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "stream": False
        }
        
        # 如果是批次的第一个请求，记录开始时间
        if self.current_batch_requests == 0:
            self.current_batch_start_time = time.time()
        
        start_time = time.time()
        try:
            async with session.post(
                self.args.url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                end_time = time.time()
                latency = (end_time - start_time) * 1000
                
                if response.status == 200:
                    result = await response.json()
                    generated_text = result.get("text", [""])[0]
                    
                    # 使用简单的分词方法估算token数量
                    tokens_generated = len(generated_text.split())
                    
                    # 更新全局计数器
                    self.request_counter += 1
                    self.successful_requests += 1
                    self.total_tokens += tokens_generated
                    
                    # 更新批次计数器
                    self.current_batch_requests += 1
                    self.current_batch_tokens += tokens_generated
                    
                    # 检查是否达到批次大小
                    if self.current_batch_requests >= self.batch_size:
                        await self.record_batch_performance()
                    
                    return {
                        "request_id": request_id,
                        "latency": latency,
                        "success": True,
                        "tokens_generated": tokens_generated
                    }
                else:
                    self.request_counter += 1
                    self.failed_requests += 1
                    self.current_batch_requests += 1
                    
                    if self.current_batch_requests >= self.batch_size:
                        await self.record_batch_performance()
                    
                    error_text = await response.text()
                    return {
                        "request_id": request_id,
                        "latency": latency,
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text[:100]}"
                    }
        except asyncio.TimeoutError:
            end_time = time.time()
            self.request_counter += 1
            self.failed_requests += 1
            self.current_batch_requests += 1
            
            if self.current_batch_requests >= self.batch_size:
                await self.record_batch_performance()
            
            return {
                "request_id": request_id,
                "latency": (end_time - start_time) * 1000,
                "success": False,
                "error": "Timeout"
            }
        except Exception as e:
            end_time = time.time()
            self.request_counter += 1
            self.failed_requests += 1
            self.current_batch_requests += 1
            
            if self.current_batch_requests >= self.batch_size:
                await self.record_batch_performance()
            
            return {
                "request_id": request_id,
                "latency": (end_time - start_time) * 1000,
                "success": False,
                "error": str(e)
            }
    
    async def request_worker(self, session, prompt, worker_id, stop_event):
        """请求工作线程"""
        request_id_base = worker_id * 1000000
        local_counter = 0
        
        while not stop_event.is_set():
            request_id = request_id_base + local_counter
            await self.send_request(session, prompt, request_id)
            local_counter += 1
            await asyncio.sleep(0.01)
    
    async def hourly_summary_collector(self, stop_event):
        """每小时汇总收集器"""
        collect_interval = 3600  # 1小时
        
        while not stop_event.is_set():
            await asyncio.sleep(collect_interval)
            
            current_time = time.time()
            elapsed_hours = (current_time - self.start_time) / 3600
            
            if self.batch_performance_data:
                # 获取最近一小时的数据
                recent_data = [d for d in self.batch_performance_data 
                             if d['elapsed_hours'] >= (elapsed_hours - 1) and d['elapsed_hours'] <= elapsed_hours]
                
                if recent_data:
                    df_hour = pd.DataFrame(recent_data)
                    
                    hourly_summary = {
                        'hour': int(elapsed_hours),
                        'timestamp': datetime.now().isoformat(),
                        'avg_rps': df_hour['batch_rps'].mean(),
                        'avg_tps': df_hour['batch_tps'].mean(),
                        'max_rps': df_hour['batch_rps'].max(),
                        'max_tps': df_hour['batch_tps'].max(),
                        'avg_gpu_temp': df_hour['gpu_temp'].mean(),
                        'max_gpu_temp': df_hour['gpu_temp'].max(),
                        'avg_power_draw': df_hour['power_draw'].mean(),
                        'max_power_draw': df_hour['power_draw'].max(),
                        'avg_gpu_util': df_hour['gpu_util'].mean(),
                        'total_requests': self.request_counter,
                        'total_tokens': self.total_tokens,
                        'success_rate': self.successful_requests / max(self.request_counter, 1)
                    }
                    
                    self.hourly_performance.append(hourly_summary)
                    
                    print(f"\n📊 第{int(elapsed_hours)}小时性能汇总:")
                    print(f"   平均RPS: {hourly_summary['avg_rps']:.2f}, 平均TPS: {hourly_summary['avg_tps']:.2f}")
                    print(f"   GPU温度: {hourly_summary['avg_gpu_temp']:.1f}°C, 功耗: {hourly_summary['avg_power_draw']:.1f}W")
                    print(f"   累计请求: {hourly_summary['total_requests']:,}, 成功率: {hourly_summary['success_rate']:.1%}")
    
    async def backup_metrics_collector(self, stop_event):
        """备用指标收集器（防止长时间没有批次完成）"""
        collect_interval = 30  # 30秒检查一次
        
        while not stop_event.is_set():
            current_time = time.time()
            
            # 如果有未完成的批次且超过一定时间没有更新，强制记录
            if (self.current_batch_requests > 0 and 
                self.last_batch_time and 
                current_time - self.last_batch_time > 10):  # 超过10秒没有完成批次
                await self.record_batch_performance()
            
            await asyncio.sleep(collect_interval)
    
    def save_batch_metrics_to_csv(self):
        """保存批次性能指标到CSV文件"""
        if not self.batch_performance_data:
            return
        
        filename = f"long_running_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            if self.batch_performance_data:
                fieldnames = self.batch_performance_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.batch_performance_data)
        
        return filename
    
    def generate_comprehensive_report(self):
        """生成综合测试报告"""
        if not self.batch_performance_data:
            print("没有收集到性能数据")
            return
        
        df = pd.DataFrame(self.batch_performance_data)
        
        # 计算整体统计信息
        total_duration_hours = (self.end_time - self.start_time) / 3600
        
        report = {
            "test_configuration": {
                "duration_hours": self.args.duration_hours,
                "concurrent_workers": self.args.concurrent_workers,
                "batch_size": self.batch_size,
                "max_tokens": self.args.max_tokens,
                "temperature": self.args.temperature,
                "url": self.args.url
            },
            "test_timing": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
                "actual_duration_hours": total_duration_hours
            },
            "request_statistics": {
                "total_requests": self.request_counter,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": self.successful_requests / max(self.request_counter, 1),
                "total_tokens": self.total_tokens
            },
            "performance_metrics": {
                "average_batch_rps": df['batch_rps'].mean(),
                "max_batch_rps": df['batch_rps'].max(),
                "average_batch_tps": df['batch_tps'].mean(),
                "max_batch_tps": df['batch_tps'].max(),
                "std_batch_rps": df['batch_rps'].std(),
                "std_batch_tps": df['batch_tps'].std(),
                "p95_batch_rps": df['batch_rps'].quantile(0.95),
                "p95_batch_tps": df['batch_tps'].quantile(0.95)
            },
            "gpu_metrics": {
                "max_gpu_temp": df['gpu_temp'].max(),
                "average_gpu_temp": df['gpu_temp'].mean(),
                "max_power_draw": df['power_draw'].max(),
                "average_power_draw": df['power_draw'].mean(),
                "max_gpu_util": df['gpu_util'].max(),
                "average_gpu_util": df['gpu_util'].mean(),
                "max_memory_util": df['memory_util'].max(),
                "average_memory_util": df['memory_util'].mean()
            },
            "system_metrics": {
                "max_cpu_util": df['cpu_percent'].max(),
                "average_cpu_util": df['cpu_percent'].mean(),
                "max_memory_util": df['memory_percent'].max(),
                "average_memory_util": df['memory_percent'].mean()
            },
            "hourly_performance": self.hourly_performance
        }
        
        # 性能稳定性分析
        stable_period = df[df['elapsed_hours'] >= 0.5]  # 排除前30分钟的热身期
        if len(stable_period) > 0:
            report["stability_analysis"] = {
                "stable_avg_rps": stable_period['batch_rps'].mean(),
                "stable_avg_tps": stable_period['batch_tps'].mean(),
                "stable_std_rps": stable_period['batch_rps'].std(),
                "stable_std_tps": stable_period['batch_tps'].std(),
                "cv_rps": stable_period['batch_rps'].std() / stable_period['batch_rps'].mean() if stable_period['batch_rps'].mean() > 0 else 0,
                "cv_tps": stable_period['batch_tps'].std() / stable_period['batch_tps'].mean() if stable_period['batch_tps'].mean() > 0 else 0
            }
        
        # 保存报告
        report_filename = f"comprehensive_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印报告摘要
        self.print_report_summary(report)
        
        return report_filename
    
    def print_report_summary(self, report):
        """打印报告摘要"""
        print("\n" + "="*80)
        print("长时间压力测试综合报告")
        print("="*80)
        
        print(f"\n📋 测试配置:")
        print(f"   持续时间: {report['test_configuration']['duration_hours']} 小时")
        print(f"   并发工作线程: {report['test_configuration']['concurrent_workers']}")
        print(f"   批次大小: {report['test_configuration']['batch_size']} 请求/批次")
        print(f"   实际运行: {report['test_timing']['actual_duration_hours']:.2f} 小时")
        
        print(f"\n📊 请求统计:")
        print(f"   总请求数: {report['request_statistics']['total_requests']:,}")
        print(f"   成功率: {report['request_statistics']['success_rate']:.2%}")
        print(f"   总生成Token: {report['request_statistics']['total_tokens']:,}")
        
        print(f"\n🚀 性能指标:")
        print(f"   平均RPS: {report['performance_metrics']['average_batch_rps']:.2f} (±{report['performance_metrics']['std_batch_rps']:.2f})")
        print(f"   最高RPS: {report['performance_metrics']['max_batch_rps']:.2f}")
        print(f"   平均TPS: {report['performance_metrics']['average_batch_tps']:.2f} (±{report['performance_metrics']['std_batch_tps']:.2f})")
        print(f"   最高TPS: {report['performance_metrics']['max_batch_tps']:.2f}")
        print(f"   P95 RPS: {report['performance_metrics']['p95_batch_rps']:.2f}")
        print(f"   P95 TPS: {report['performance_metrics']['p95_batch_tps']:.2f}")
        
        if 'stability_analysis' in report:
            stable = report['stability_analysis']
            print(f"   稳定期RPS: {stable['stable_avg_rps']:.2f} (±{stable['stable_std_rps']:.2f}, CV: {stable['cv_rps']:.3f})")
            print(f"   稳定期TPS: {stable['stable_avg_tps']:.2f} (±{stable['stable_std_tps']:.2f}, CV: {stable['cv_tps']:.3f})")
        
        print(f"\n🔥 GPU指标:")
        print(f"   温度 - 平均: {report['gpu_metrics']['average_gpu_temp']:.1f}°C, 最高: {report['gpu_metrics']['max_gpu_temp']:.1f}°C")
        print(f"   功耗 - 平均: {report['gpu_metrics']['average_power_draw']:.1f}W, 最高: {report['gpu_metrics']['max_power_draw']:.1f}W")
        print(f"   使用率 - 平均: {report['gpu_metrics']['average_gpu_util']:.1f}%, 最高: {report['gpu_metrics']['max_gpu_util']:.1f}%")
        
        print(f"\n💻 系统指标:")
        print(f"   CPU使用率 - 平均: {report['system_metrics']['average_cpu_util']:.1f}%, 最高: {report['system_metrics']['max_cpu_util']:.1f}%")
        print(f"   内存使用率 - 平均: {report['system_metrics']['average_memory_util']:.1f}%, 最高: {report['system_metrics']['max_memory_util']:.1f}%")
        
        # 每小时性能趋势
        if self.hourly_performance:
            print(f"\n📈 每小时性能趋势:")
            for hour_data in self.hourly_performance:
                print(f"   第{hour_data['hour']}小时: RPS={hour_data['avg_rps']:.2f}, TPS={hour_data['avg_tps']:.2f}, "
                      f"GPU={hour_data['avg_gpu_temp']:.1f}°C/{hour_data['avg_gpu_util']:.1f}%")
    
    async def run(self):
        """运行长时间压力测试"""
        print(f"🚀 开始长时间压力测试")
        print(f"⏱️  持续时间: {self.args.duration_hours} 小时")
        print(f"👥 并发工作线程: {self.args.concurrent_workers}")
        print(f"📦 批次大小: {self.batch_size} 请求/批次")
        print(f"🌐 目标URL: {self.args.url}")
        print(f"📝 最大生成长度: {self.args.max_tokens} tokens")
        print(f"🌡️  温度: {self.args.temperature}")
        print(f"🕐 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        print("批次 | 瞬时RPS | 瞬时TPS | 平均RPS | 平均TPS | GPU状态 | 功耗 | 成功率")
        print("-" * 80)
        
        self.start_time = time.time()
        self.current_batch_start_time = self.start_time
        self.last_batch_time = self.start_time
        
        # 生成测试提示
        test_prompt = self.args.prompt
        if self.args.prompt_length > 0:
            test_prompt += " " + "测试文本" * (self.args.prompt_length // 4)
        
        stop_event = asyncio.Event()
        
        try:
            async with aiohttp.ClientSession() as session:
                # 启动备用指标收集器
                backup_task = asyncio.create_task(self.backup_metrics_collector(stop_event))
                
                # 启动每小时汇总收集器
                hourly_task = asyncio.create_task(self.hourly_summary_collector(stop_event))
                
                # 启动工作线程
                worker_tasks = []
                for i in range(self.args.concurrent_workers):
                    task = asyncio.create_task(self.request_worker(session, test_prompt, i, stop_event))
                    worker_tasks.append(task)
                
                # 运行指定时长
                await asyncio.sleep(self.args.duration_hours * 3600)
                
                # 停止测试前记录最后一个批次
                if self.current_batch_requests > 0:
                    await self.record_batch_performance()
                
                print("\n" + "="*80)
                print("⏹️  测试时间到，正在停止测试...")
                stop_event.set()
                
                # 等待所有任务结束
                await asyncio.gather(*worker_tasks, return_exceptions=True)
                await backup_task
                await hourly_task
            
            self.end_time = time.time()
            
            # 生成最终报告
            report_filename = self.generate_comprehensive_report()
            
            # 保存性能数据文件路径
            csv_filename = self.save_batch_metrics_to_csv()
            print(f"\n💾 性能数据已保存到: {csv_filename}")
            print(f"📄 详细报告已保存到: {report_filename}")
            
        except KeyboardInterrupt:
            print("\n⏹️  用户中断测试，正在停止...")
            if self.current_batch_requests > 0:
                await self.record_batch_performance()
            stop_event.set()
            self.end_time = time.time()
            report_filename = self.generate_comprehensive_report()
            print(f"📄 详细报告已保存到: {report_filename}")

def main():
    parser = argparse.ArgumentParser(description='vLLM 长时间压力测试工具 - 最终版')
    parser.add_argument('--duration-hours', type=float, default=3.0, help='测试持续时间（小时）')
    parser.add_argument('--concurrent-workers', type=int, default=20, help='并发工作线程数')
    parser.add_argument('--batch-size', type=int, default=10, help='每批请求数量')
    parser.add_argument('--prompt-length', type=int, default=256, help='提示词长度')
    parser.add_argument('--max-tokens', type=int, default=500, help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.7, help='温度参数')
    parser.add_argument('--url', type=str, default='http://localhost:8000/generate', help='vLLM服务器URL')
    parser.add_argument('--prompt', type=str, default='请解释人工智能的基本原理和应用领域。', help='测试提示词')
    
    args = parser.parse_args()
    
    benchmark = FinalLongRunningBenchmark(args)
    asyncio.run(benchmark.run())

if __name__ == "__main__":
    main()
