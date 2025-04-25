#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器连接模块 - 用于SSH连接和远程操作
该模块提供SSH服务器连接、文件上传和远程命令执行功能。
"""

import os
import paramiko
from scp import SCPClient
import time
import streamlit as st

class ServerConnection:
    """SSH服务器连接和操作类"""
    
    def __init__(self):
        """初始化服务器连接类"""
        self.client = None
        self.scp = None
        self.connected = False
        self.host = None
        self.port = None
        self.username = None
        
    def connect(self, host, port, username, auth_method="password", password=None, key_path=None):
        """
        连接到SSH服务器
        
        参数:
            host (str): 服务器主机地址
            port (int): 服务器端口
            username (str): 用户名
            auth_method (str): 认证方式，"password"或"key"
            password (str, 可选): 密码，当auth_method为"password"时使用
            key_path (str, 可选): 密钥文件路径，当auth_method为"key"时使用
            
        返回:
            bool: 连接是否成功
            str: 成功或错误消息
        """
        try:
            # 创建SSH客户端
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 根据认证方式连接
            if auth_method == "password":
                self.client.connect(
                    hostname=host,
                    port=port,
                    username=username,
                    password=password,
                    timeout=10
                )
            elif auth_method == "key":
                if key_path and os.path.exists(key_path):
                    key = paramiko.RSAKey.from_private_key_file(key_path)
                    self.client.connect(
                        hostname=host,
                        port=port,
                        username=username,
                        pkey=key,
                        timeout=10
                    )
                else:
                    return False, "密钥文件不存在"
            else:
                return False, "不支持的认证方式"
            
            # 创建SCP客户端
            self.scp = SCPClient(self.client.get_transport())
            
            # 保存连接信息
            self.connected = True
            self.host = host
            self.port = port
            self.username = username
            
            return True, "成功连接到服务器"
            
        except paramiko.AuthenticationException:
            return False, "认证失败，请检查用户名和密码/密钥"
        except paramiko.SSHException as e:
            return False, f"SSH连接错误: {str(e)}"
        except Exception as e:
            return False, f"连接错误: {str(e)}"
    
    def disconnect(self):
        """断开服务器连接"""
        if self.scp:
            self.scp.close()
            self.scp = None
        
        if self.client:
            self.client.close()
            self.client = None
        
        self.connected = False
        return "已断开服务器连接"
    
    def upload_file(self, local_path, remote_path):
        """
        上传文件到服务器
        
        参数:
            local_path (str): 本地文件路径
            remote_path (str): 远程文件路径
            
        返回:
            bool: 上传是否成功
            str: 成功或错误消息
        """
        if not self.connected or not self.client:
            return False, "未连接到服务器"
        
        try:
            if not os.path.exists(local_path):
                return False, "本地文件不存在"
            
            self.scp.put(local_path, remote_path)
            return True, f"文件已上传到 {remote_path}"
        except Exception as e:
            return False, f"文件上传错误: {str(e)}"
    
    def execute_command(self, command, timeout=60):
        """
        在服务器上执行命令
        
        参数:
            command (str): 要执行的命令
            timeout (int): 命令超时时间（秒）
            
        返回:
            bool: 执行是否成功
            dict: 包含stdout、stderr和exit_code的结果字典
        """
        if not self.connected or not self.client:
            return False, {"error": "未连接到服务器"}
        
        try:
            # 执行命令
            stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
            
            # 获取结果
            exit_code = stdout.channel.recv_exit_status()
            stdout_str = stdout.read().decode('utf-8')
            stderr_str = stderr.read().decode('utf-8')
            
            result = {
                "stdout": stdout_str,
                "stderr": stderr_str,
                "exit_code": exit_code
            }
            
            return exit_code == 0, result
        except Exception as e:
            return False, {"error": f"命令执行错误: {str(e)}"}
    
    def start_training(self, script_path, args=None, background=True):
        """
        在服务器上启动训练任务
        
        参数:
            script_path (str): 脚本路径
            args (str, 可选): 脚本参数
            background (bool): 是否在后台运行
            
        返回:
            bool: 启动是否成功
            dict: 结果信息
        """
        if not self.connected or not self.client:
            return False, {"error": "未连接到服务器"}
        
        try:
            # 构建命令
            command = f"python {script_path}"
            if args:
                command += f" {args}"
            
            if background:
                # 在后台运行并将输出重定向到日志文件
                log_file = f"{script_path}.log"
                command = f"nohup {command} > {log_file} 2>&1 &"
                
                # 执行命令
                stdin, stdout, stderr = self.client.exec_command(command)
                time.sleep(1)  # 等待命令启动
                
                return True, {"message": f"训练任务已在后台启动，日志保存在 {log_file}"}
            else:
                # 在前台运行
                return self.execute_command(command, timeout=3600)  # 1小时超时
        except Exception as e:
            return False, {"error": f"启动训练任务错误: {str(e)}"}
    
    def check_status(self):
        """
        检查服务器状态
        
        返回:
            bool: 检查是否成功
            dict: 状态信息
        """
        if not self.connected or not self.client:
            return False, {"error": "未连接到服务器"}
        
        try:
            # 获取系统信息
            success, result = self.execute_command("uptime && free -h && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv")
            
            if success:
                return True, {
                    "uptime": result["stdout"].split("\n")[0],
                    "memory": result["stdout"].split("\n")[1:3],
                    "gpu": result["stdout"].split("\n")[3:]
                }
            else:
                return False, {"error": "获取服务器状态失败", "details": result}
        except Exception as e:
            return False, {"error": f"检查状态错误: {str(e)}"}
