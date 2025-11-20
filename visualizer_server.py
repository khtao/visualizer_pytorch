# app.py
import os
import sys
import argparse
from pathlib import Path
from flask import Flask, render_template, send_from_directory, jsonify, request, abort
from flask_socketio import SocketIO
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='files')
parser.add_argument('--host', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=5000)
args = parser.parse_args()

ROOT_DIR = Path(args.root).resolve()
if not ROOT_DIR.exists():
    print(f"❌ 根目录不存在: {ROOT_DIR}")
    sys.exit(1)

print(f"📁 文件根目录: {ROOT_DIR}")
print(f"🌐 访问地址: http://{args.host}:{args.port}")

app = Flask(__name__, static_folder=None)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app, async_mode='threading')

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
TEXT_EXTENSIONS = {'.txt', '.log', '.md', '.csv', '.json', '.xml', '.yml', '.yaml', '.ini', '.cfg'}


def is_image_file(filename):
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def is_text_file(filename):
    return Path(filename).suffix.lower() in TEXT_EXTENSIONS


def safe_join(base: Path, *paths) -> Path:
    joined = base.joinpath(*paths).resolve()
    if not str(joined).startswith(str(base)):
        raise ValueError("Invalid path traversal.")
    return joined


# === 全局状态 ===
current_project_lock = threading.Lock()
current_project = None

observers = {}
debounce_timers = {}


class ProjectImageChangeHandler(FileSystemEventHandler):
    def __init__(self, project_name):
        self.project_name = project_name

    def on_any_event(self, event):
        if event.is_directory:
            return
        if not is_image_file(event.src_path):
            return  # 忽略非图像
        if any(event.src_path.endswith(s) for s in ('.swp', '~', '.tmp', '.DS_Store')):
            return

        with current_project_lock:
            if current_project != self.project_name:
                return

        # 防抖：300ms 后触发一次
        timer = debounce_timers.get(self.project_name)
        if timer:
            timer.cancel()
        new_timer = threading.Timer(0.3, self._emit_update)
        debounce_timers[self.project_name] = new_timer
        new_timer.start()

    def _emit_update(self):
        with current_project_lock:
            if current_project != self.project_name:
                return
        socketio.emit('image_update', {'project': self.project_name})


def start_observer(project_name):
    if project_name in observers:
        return
    proj_path = safe_join(ROOT_DIR, project_name)
    if not proj_path.is_dir():
        return
    handler = ProjectImageChangeHandler(project_name)
    observer = Observer()
    observer.schedule(handler, str(proj_path), recursive=True)
    observer.start()
    observers[project_name] = observer
    print(f"🔍 监控图像变动: {project_name}")


def stop_observer(project_name):
    obs = observers.pop(project_name, None)
    if obs:
        obs.stop()
        obs.join()
    timer = debounce_timers.pop(project_name, None)
    if timer:
        timer.cancel()


# === 路由 ===

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/projects')
def list_projects():
    try:
        projects = sorted([f.name for f in ROOT_DIR.iterdir() if f.is_dir()])
        return jsonify(projects=projects)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/api/set_current_project/<project>')
def set_current_project(project):
    global current_project
    try:
        proj_path = safe_join(ROOT_DIR, project)
        if not proj_path.is_dir():
            return jsonify(ok=False, error="Project not found"), 404
        with current_project_lock:
            old = current_project
            current_project = project
        if old and old != project:
            stop_observer(old)
        start_observer(project)
        return jsonify(ok=True)
    except ValueError:
        return jsonify(ok=False, error="Invalid project"), 403
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route('/api/albums/<project>')
def list_albums(project):
    try:
        proj_path = safe_join(ROOT_DIR, project)
        if not proj_path.is_dir():
            return jsonify(albums=[], is_flat=True)
        subdirs = [f.name for f in proj_path.iterdir() if f.is_dir()]
        if not subdirs:
            return jsonify(albums=[], is_flat=True)
        else:
            return jsonify(albums=sorted(subdirs), is_flat=False)
    except ValueError:
        abort(403)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/api/files/<project>/<path:subpath>')
def get_files(project, subpath=''):
    try:
        proj_path = safe_join(ROOT_DIR, project)
        if subpath == '-':
            target = proj_path
        else:
            target = safe_join(proj_path, subpath)
        if not target.is_dir():
            abort(404)
        files = [f.name for f in target.iterdir() if f.is_file()]
        images = sorted([f for f in files if is_image_file(f)])
        texts = sorted([f for f in files if is_text_file(f)])
        return jsonify(images=images, texts=texts)
    except ValueError:
        abort(403)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/static/files/<path:filepath>')
def serve_file(filepath):
    try:
        full = safe_join(ROOT_DIR, filepath)
        if not full.is_file():
            abort(404)
        return send_from_directory(full.parent, full.name)
    except ValueError:
        abort(403)


# === 根目录监控（项目增删）===
class RootHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            projects = sorted([f.name for f in ROOT_DIR.iterdir() if f.is_dir()])
            socketio.emit('project_list_update', {'projects': projects})


root_obs = Observer()
root_obs.schedule(RootHandler(), str(ROOT_DIR), recursive=False)
root_obs.start()


def cleanup():
    root_obs.stop()
    for p in list(observers.keys()):
        stop_observer(p)
    root_obs.join()


if __name__ == '__main__':
    try:
        socketio.run(app, host=args.host, port=args.port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
