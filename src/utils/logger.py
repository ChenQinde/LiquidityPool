# logger_config.py
import logging
import os
from typing import Dict

class Logger:
    _instance = None
    _initialized = False
    _log_file = None
    _console_output = None  # 新增：保存控制台输出设置

    @classmethod
    def setup(cls, log_file: str = 'trading.log', console_output: bool = True):
        """
        初始化日志配置，只需在主程序中调用一次
        """
        # 保存配置
        cls._log_file = os.path.abspath(log_file)
        cls._console_output = console_output
        
        if not cls._initialized:
            logger = logging.getLogger('trading_system')
            logger.setLevel(logging.DEBUG)
            
            # 清除可能存在的处理器
            if logger.handlers:
                logger.handlers.clear()
            
            # 创建文件处理器
            file_handler = logging.FileHandler(cls._log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # 根据设置决定是否添加控制台输出
            if cls._console_output:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.DEBUG)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            cls._instance = logger
            cls._initialized = True
            return logger
    
    @classmethod
    def get_logger(cls):
        """
        获取 logger 实例。如果还没初始化，将使用主程序中的设置
        """
        if cls._initialized:
            return cls._instance
        
        # 如果主程序还没有初始化，则不使用默认值，而是等待主程序的设置
        raise RuntimeError(
            "Logger not initialized. Please call Logger.setup() in main program first."
        )

    @classmethod
    def is_initialized(cls):
        """
        检查 logger 是否已经初始化
        """
        return cls._initialized