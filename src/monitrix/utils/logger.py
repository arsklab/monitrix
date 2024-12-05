# Copyright (C) 2024 Arisaka Naoya
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging.config
import io
import yaml
from typing import Optional, Literal, Dict
from pathlib import Path

CONFIG = """
version: 1
disable_existing_loggers: False  # これを追加
formatters:
    brief:
        format: '%(message)s'
    simple:
        format: '%(asctime)s [%(levelname)-8s] : %(message)s'
    default:
        format: '%(asctime)s [%(levelname)-8s] %(name)s, Func %(funcName)s, Line %(lineno)d : %(message)s'
    thread:
        format: '%(asctime)s <%(name)s> [%(levelname)-8s] %(processName)s-%(threadName)s %(module)s-%(funcName)s-%(lineno)d : %(message)s'
handlers:
    console:
        class: logging.StreamHandler
        level: WARNING
        formatter: simple
    rotating_file:
        class: logging.handlers.RotatingFileHandler
        filename: {error_log_file_path}
        encoding: utf-8
        maxBytes: 1048576
        backupCount: 5
        level: WARNING
        formatter: default
loggers:
    {logger_name}:
        level: {level}
        handlers: [console,  rotating_file]
        propagate: False
"""

NULL_CONFIG = """
version: 1
handlers:
    null:
        class: logging.NullHandler
loggers:
    {logger_name}:
        level: {level}
        handlers: [null]
        propagate: False
"""

LEVEL = Literal["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# ロガーのキャッシュを保持
_logger_cache: Dict[str, logging.Logger] = {}

def get_logger(
    name: Optional[str] = None,
    level: LEVEL = "WARNING",
    log_dir: Optional[str] = None,
    *,
    debug_log_fname:str="debug.log",
    error_log_fname:str="error.log",
    null_logger: bool = False,
    force_new: bool = False
) -> logging.Logger:
    """ロガーの取得"""
    _name: str = __name__ if name is None else name
    
    # キャッシュされたロガーがあれば返す
    if not force_new and _name in _logger_cache:
        return _logger_cache[_name]
    
    # configの読み込み
    config_text = NULL_CONFIG if null_logger else CONFIG
    
    # ファイルの保存先
    if log_dir is None:
        # カレントディレクトリを基準にする
        log_dir_path = Path.cwd() / "logs"
    else:
        log_dir_path = Path(log_dir)
    
    # ログディレクトリの作成
    if not log_dir_path.is_dir():
      log_dir_path.mkdir(parents=True, exist_ok=True)
      print(f"ログディレクトリの作成:{log_dir_path.as_posix()}")

    try:
        # 設定の適用
        config_dict = yaml.safe_load(
            io.StringIO(
                config_text.format(
                    logger_name=_name,
                    level=level,
                    debug_log_file_path=str((log_dir_path/debug_log_fname).as_posix()),
                    error_log_file_path=str((log_dir_path/error_log_fname).as_posix()),
                )
            )
        )
        logging.config.dictConfig(config_dict)
        
        # ロガーの取得
        logger = logging.getLogger(_name)
        
        # キャッシュに保存
        _logger_cache[_name] = logger
        
        return logger
    
    except Exception as e:
        # エラーが発生した場合は基本的なロガーを返す
        print(f"Logger configuration failed: {e}")
        fallback_logger = logging.getLogger(_name)
        fallback_logger.setLevel(getattr(logging, level))
        if not fallback_logger.handlers:
            fallback_logger.addHandler(logging.StreamHandler())
        return fallback_logger