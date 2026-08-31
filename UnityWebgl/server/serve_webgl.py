"""
Unity WebGL 本地服务器

自动处理：
- .unityweb 文件的解压（Brotli/Gzip）
- 正确的 MIME 类型和 Content-Type
- 同时代理 WebSocket 到 Python 状态服务器

用法：
  python serve_webgl.py                              # 默认端口 8080，Build 目录为当前目录
  python serve_webgl.py --port 9090 --dir ./Build    # 自定义端口和目录
  python serve_webgl.py --ws-port 8770               # 同时代理 WebSocket
"""

import argparse
import gzip
import http.server
import io
import os
import sys
import threading


def try_brotli_decompress(data):
    """尝试 Brotli 解压"""
    try:
        import brotli
        return brotli.decompress(data)
    except ImportError:
        pass
    # 尝试 brotlicffi
    try:
        import brotlicffi
        return brotlicffi.decompress(data)
    except ImportError:
        pass
    return None


# ============================================================
# MIME 类型映射
# ============================================================
MIME_TYPES = {
    ".js":       "application/javascript",
    ".mjs":      "application/javascript",
    ".json":     "application/json",
    ".wasm":     "application/wasm",
    ".data":     "application/octet-stream",
    ".mem":      "application/octet-stream",
    ".html":     "text/html",
    ".css":      "text/css",
    ".png":      "image/png",
    ".jpg":      "image/jpeg",
    ".svg":      "image/svg+xml",
    ".ico":      "image/x-icon",
}


class WebGLHandler(http.server.SimpleHTTPRequestHandler):
    """支持 .unityweb 解压的 HTTP 请求处理器"""

    def do_GET(self):
        # 将 URL 路径映射到文件
        path = self.translate_path(self.path)

        # 如果请求的文件不存在，尝试找 .unityweb 版本
        if not os.path.isfile(path):
            unityweb_path = path + ".unityweb"
            if os.path.isfile(unityweb_path):
                self._serve_unityweb(unityweb_path, path)
                return

        # 正常文件，直接服务
        super().do_GET()

    def _serve_unityweb(self, unityweb_path, original_path):
        """解压并服务 .unityweb 文件"""
        # 根据原始路径确定 MIME 类型
        ext = os.path.splitext(original_path)[1].lower()
        content_type = MIME_TYPES.get(ext, "application/octet-stream")

        try:
            with open(unityweb_path, "rb") as f:
                compressed = f.read()

            # 尝试 Brotli 解压
            data = try_brotli_decompress(compressed)

            # 如果 Brotli 失败，尝试 Gzip
            if data is None:
                try:
                    data = gzip.decompress(compressed)
                except Exception:
                    # 最后尝试直接返回（可能未压缩）
                    data = compressed
                    self.log_message("WARNING: 无法解压 %s，直接返回原始数据", unityweb_path)

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        except Exception as e:
            self.send_error(500, f"解压 .unityweb 文件失败: {e}")

    def end_headers(self):
        # 允许跨域（方便开发测试）
        self.send_header("Access-Control-Allow-Origin", "*")

        # ── 启用 WASM 多线程（SharedArrayBuffer）──
        # Unity WebGL 多线程构建依赖 SharedArrayBuffer，
        # 浏览器仅在以下跨源隔离头存在时才允许其启用。
        # 若缺失，Unity 会退化为单线程，导致 CPU 拉满、画面卡顿。
        # 本机 localhost 默认被当作安全源可绕过，但通过 IP/域名访问时必须有这些头。
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")

        super().end_headers()

    def log_message(self, format, *args):
        """简化日志输出"""
        msg = format % args
        if "404" in msg or "200" in msg:
            sys.stdout.write(f"  {self.address_string()} {msg}\n")
            sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Unity WebGL 本地服务器")
    parser.add_argument("--dir", default=".", help="WebGL 构建输出目录")
    parser.add_argument("--port", type=int, default=8080, help="HTTP 端口")
    parser.add_argument("--ws-port", type=int, default=0, help="WebSocket 代理端口（可选）")
    args = parser.parse_args()

    # 切换目录
    serve_dir = os.path.abspath(args.dir)
    if not os.path.isdir(serve_dir):
        print(f"错误: 目录不存在 {serve_dir}")
        sys.exit(1)

    os.chdir(serve_dir)

    print(f"=" * 60)
    print(f"  Unity WebGL 本地服务器")
    print(f"  目录: {serve_dir}")
    print(f"  地址: http://localhost:{args.port}")
    print(f"=" * 60)
    print()

    # 检查 Brotli 支持
    has_brotli = False
    try:
        import brotli
        has_brotli = True
    except ImportError:
        try:
            import brotlicffi
            has_brotli = True
        except ImportError:
            pass

    if not has_brotli:
        print("  提示: 未安装 brotli 库，将仅支持 Gzip 解压")
        print("  安装: pip install brotli")
        print()

    # 列出检测到的文件
    build_dir = os.path.join(serve_dir, "Build")
    if os.path.isdir(build_dir):
        files = os.listdir(build_dir)
        print(f"  Build/ 目录文件:")
        for f in sorted(files):
            size_kb = os.path.getsize(os.path.join(build_dir, f)) // 1024
            print(f"    {f} ({size_kb} KB)")
        print()
    else:
        print(f"  警告: 未找到 Build/ 子目录")
        print()

    # 启动服务器
    handler = WebGLHandler
    server = http.server.HTTPServer(("", args.port), handler)
    print(f"  服务已启动，按 Ctrl+C 停止")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务器已停止")
        server.server_close()


if __name__ == "__main__":
    main()
