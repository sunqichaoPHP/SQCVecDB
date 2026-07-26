#!/usr/bin/env python3
"""
启动 xiangliang REST API 服务

用法:
    python examples/run_service.py [--port 8000] [--data-dir ./xiangliang_data]
"""

import argparse
import uvicorn
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="启动 xiangliang REST API 服务")
    parser.add_argument("--port", type=int, default=8000, help="服务端口 (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="服务地址 (default: 127.0.0.1)")
    parser.add_argument("--data-dir", default="./xiangliang_data", help="数据目录")
    parser.add_argument("--reload", action="store_true", help="启用热重载")
    
    args = parser.parse_args()
    
    # 创建数据目录
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 启动 xiangliang REST API 服务")
    print(f"   地址: http://{args.host}:{args.port}")
    print(f"   数据目录: {args.data_dir}")
    print(f"   文档: http://{args.host}:{args.port}/docs")
    print()
    
    uvicorn.run(
        "xiangliang.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
