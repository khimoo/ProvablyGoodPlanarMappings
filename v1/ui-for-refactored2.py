import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.widgets import Slider, Button
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any
from refactored2 import BetterFitwithGaussianRBF
from scipy.spatial import Delaunay


# ----------------- 基底クラス -----------------
class GraphUI(ABC):
    def __init__(self,
                 G: nx.Graph,
                 initial_pos: Dict[Any, Tuple[float, float]],
                 figsize: Tuple[int,int]=(6,6)):
        self.G = G
        self.pos = initial_pos.copy()
        self.node_list: List[Any] = list(self.G.nodes)

        # 図を作る
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # ウィジェット用の領域はそのままにして、メインのグラフ領域を広く確保する。
        # set_position([left, bottom, width, height]) の値は (0..1) の比率。
        # 下側にスライダ等のスペースを確保しつつ、グラフを大きくする。
        self.ax.set_position([0.05, 0.18, 1.90, 0.78])  # ← 高さを広げたい場合は height (最後の値) を増やす

        # 既存コードと整合させるために plt.subplots_adjust も軽く指定（ウィジェットが下に収まるように）
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05)

        # 以下は既存の描画初期化
        self.nodes_pc = nx.draw_networkx_nodes(self.G, pos=self.pos, ax=self.ax, node_color="c", node_size=30)
        self.edge_lc = nx.draw_networkx_edges(self.G, pos=self.pos, ax=self.ax)
        self.labels_dict = nx.draw_networkx_labels(self.G, pos=self.pos, ax=self.ax)

        self.ax.set_aspect('equal')
        self.ax.set_title("Progressive Animation GraphUI")
        self.ax.axis('off')

        self.state: Any = None
        self.widgets: Dict[str, Any] = {}


    @abstractmethod
    def init_state(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def create_ui(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, val=None):
        raise NotImplementedError

    def start(self):
        self.state = self.init_state()
        self.create_ui()
        reset_btn = self.widgets.get('reset_button', None)
        if reset_btn is not None:
            reset_btn.on_clicked(self._on_reset_default)
        self.update(None)
        plt.show()

    def _on_reset_default(self, event):
        # 単純に初期座標に戻す
        self.pos.update(self.initial_pos)
        if isinstance(self.state, dict):
            if 'a_key' in self.state and 'a_pos' in self.state:
                self.state['a_pos'] = self.initial_pos[self.state['a_key']]
        self.update(None)


# ----------------- 子クラス：逐次計算・即時表示 -----------------
class ProgressiveAnimationGraphUI(GraphUI):
    def __init__(self, G, initial_pos, a_key, src_ctrl=None, **kwargs):
        super().__init__(G, initial_pos, **kwargs)
        self.initial_pos = initial_pos.copy()
        self.state = {}
        self.a_key = a_key

        # 制御点（src）を外部から与えられれば使う。なければデフォルトを用意。
        if src_ctrl is None:
            # デフォルト制御点（例）
            self.src_ctrl = np.array([[1.0, 10.0], [13.0, 9.0], [20.0, 9.0], [20.0, 10.0]], dtype=float)
        else:
            self.src_ctrl = np.asarray(src_ctrl, dtype=float)

        # アニメーション中に更新される dst 制御点を保持
        self.current_dst_ctrl = self.src_ctrl.copy()
        # 可視化アーティスト参照（create_uiで初期化）
        self.ctrl_src_scatter = None
        self.ctrl_dst_scatter = None
        self.ctrl_src_texts = []
        self.ctrl_dst_texts = []
        self.center_marker = None

    def init_state(self):
        return {
            'a_pos': self.initial_pos[self.a_key],
            'steps': 30,
            'digree': 180.0,
            # 新しく UI で操作できるパラメータ
            'K_threshold': 2.0,
            'fps_k': 3,
            # 'frames': []
        }

    def create_ui(self):
        # スライダー領域
        ax_steps = plt.axes([0.12, 0.22, 0.76, 0.03])
        ax_digree = plt.axes([0.12, 0.17, 0.76, 0.03])
        ax_k = plt.axes([0.12, 0.12, 0.76, 0.03])
        ax_fps = plt.axes([0.12, 0.07, 0.76, 0.03])

        # ボタン領域（stop削除後、computeとresetのみ）
        ax_compute = plt.axes([0.02, 0.44, 0.10, 0.05])
        ax_reset = plt.axes([0.02, 0.37, 0.10, 0.05])

        slider_steps = Slider(ax_steps, "step (frames)", 1, 60, valinit=self.state['steps'], valstep=1)
        slider_digree = Slider(ax_digree, "digree (°)", -360.0, 360.0, valinit=self.state['digree'])
        slider_k = Slider(ax_k, "K_threshold", 1, 20, valinit=self.state['K_threshold'], valstep=0.1)
        slider_fps = Slider(ax_fps, "fps_k", 1, 200, valinit=self.state['fps_k'], valstep=1)

        btn_compute = Button(ax_compute, "Start")
        btn_reset = Button(ax_reset, "Reset")

        self.widgets.update({
            'slider_steps': slider_steps,
            'slider_digree': slider_digree,
            'slider_k': slider_k,
            'slider_fps': slider_fps,
            'compute_button': btn_compute,
            'reset_button': btn_reset
        })

        self.status_text = self.ax.text(0.5, 1.02, "", transform=self.ax.transAxes,
                                        ha='center', va='bottom', fontsize=12,
                                        color='red', visible=False)

        slider_steps.on_changed(self._on_slider_change)
        slider_digree.on_changed(self._on_slider_change)
        slider_k.on_changed(self._on_slider_change)
        slider_fps.on_changed(self._on_slider_change)
        btn_compute.on_clicked(self._on_compute)
        btn_reset.on_clicked(lambda ev: self.reset_positions())

        self.is_computing = False


        # ------------------------------
        # 制御点の視覚化を初期化
        # ------------------------------
        # src 制御点：丸、大きめ、縁取り
        self.ctrl_src_scatter = self.ax.scatter(
            self.src_ctrl[:, 0], self.src_ctrl[:, 1],
            s=120, marker='o', edgecolors='black', linewidths=0.8, zorder=5, label="src_ctrl"
        )
        # dst 制御点（初期は src と同じ位置）：三角で表示（アニメーションで移動）
        self.ctrl_dst_scatter = self.ax.scatter(
            self.current_dst_ctrl[:, 0], self.current_dst_ctrl[:, 1],
            s=100, marker='^', edgecolors='black', linewidths=0.8, zorder=6, label="dst_ctrl"
        )

        # 制御点ラベル（index）を表示（src）と（dst）
        for i, (x, y) in enumerate(self.src_ctrl):
            txt = self.ax.text(x, y, f"S{i}", fontsize=9, fontweight='bold', va='bottom', ha='right', zorder=7)
            self.ctrl_src_texts.append(txt)
        for i, (x, y) in enumerate(self.current_dst_ctrl):
            txt = self.ax.text(x, y, f"D{i}", fontsize=9, va='top', ha='left', zorder=8)
            self.ctrl_dst_texts.append(txt)

        # --- activated 用のアーティストを追加 ---
        self.activated_scatter = self.ax.scatter(
            [], [], s=100, marker='o', facecolors='none',
            edgecolors='red', linewidths=1.4, zorder=10, label="activated"
        )

        # --- farthest_points 用のアーティスト（青の ×）を追加 ---
        self.farthest_scatter = self.ax.scatter(
            [], [], s=80, marker='X', facecolors='none',
            edgecolors='blue', linewidths=1.2, zorder=11, label="farthest"
        )

        # 凡例（必要なら） - 既存の legend 呼び出しの後に再描画
        try:
            self.ax.legend(loc='upper left')
        except Exception:
            pass


    def _set_buttons_visible(self, compute_visible: bool):
        btn_c = self.widgets['compute_button']
        btn_c.ax.set_visible(compute_visible)
        self.fig.canvas.draw_idle()


    def _show_status(self, msg: str):
        self.status_text.set_text(msg)
        self.status_text.set_visible(True)
        self.fig.canvas.draw_idle()

    def _hide_status(self):
        self.status_text.set_visible(False)
        self.fig.canvas.draw_idle()

    def _on_slider_change(self, val):
        if getattr(self, "is_computing", False):
            return
        self.state['steps'] = int(self.widgets['slider_steps'].val)
        self.state['digree'] = float(self.widgets['slider_digree'].val)
        self.state['K_threshold'] = float(self.widgets['slider_k'].val)
        self.state['fps_k'] = int(self.widgets['slider_fps'].val)

    def _update_ctrl_artists(self, dst_ctrl: np.ndarray):
        """dst_ctrl: (k,2) ndarray"""
        # 更新：dst scatter
        self.current_dst_ctrl = dst_ctrl.copy()
        self.ctrl_dst_scatter.set_offsets(self.current_dst_ctrl)

        # dst ラベル更新（位置を更新。テキスト数は固定）
        for i, txt in enumerate(self.ctrl_dst_texts):
            if i < len(self.current_dst_ctrl):
                txt.set_position(self.current_dst_ctrl[i])
            else:
                txt.set_position((np.nan, np.nan))

        # src は固定
        self.ctrl_src_scatter.set_offsets(self.src_ctrl)
        for i, txt in enumerate(self.ctrl_src_texts):
            txt.set_position(self.src_ctrl[i])

    def _update_activated_artist(self, activated_indices: List[int]):
        """activated_indices の頂点位置に合わせて activated_scatter を更新する。"""
        if not activated_indices:
            self.activated_scatter.set_offsets(np.empty((0, 2)))
        else:
            pts = np.array([self.pos[int(idx)] for idx in activated_indices])
            self.activated_scatter.set_offsets(pts)
        self.fig.canvas.draw_idle()

    def _update_farthest_artist(self, farthest_indices):
        """farthest_points のインデックスに合わせて farthest_scatter を更新する。
           farthest_indices は None / [] / iterable of indices を想定。
        """
        if farthest_indices is None:
            self.farthest_scatter.set_offsets(np.empty((0, 2)))
            self.fig.canvas.draw_idle()
            return

        # もし boolean mask が来たらインデックスに変換
        if isinstance(farthest_indices, (np.ndarray, list)) and getattr(farthest_indices, "dtype", None) == bool:
            farthest_indices = np.nonzero(farthest_indices)[0]

        # 空チェック
        if len(farthest_indices) == 0:
            self.farthest_scatter.set_offsets(np.empty((0, 2)))
        else:
            pts = np.array([self.pos[int(idx)] for idx in farthest_indices])
            self.farthest_scatter.set_offsets(pts)
        self.fig.canvas.draw_idle()

    def _on_compute(self, event):
        if self.is_computing:
            print("すでに計算中です。")
            return

        self.is_computing = True
        self._set_buttons_visible(False)
        self._show_status("Now Computing...")

        try:
            steps = self.state['steps']
            digree = self.state['digree']
            a0 = np.array(self.state['a_pos'])
            K_threshold = float(self.state.get('K_threshold', 2))
            fps_k = int(self.state.get('fps_k', 100))

            center = [35.0,10.0]
            a0_rel = a0 - center
            r = np.linalg.norm(a0_rel)

            if r == 0:
                for _ in range(steps):
                    frame = tuple(a0)
                    self.pos[self.a_key] = frame
                    self._update_artists()
                    # 両方クリア
                    self._update_activated_artist([])
                    self._update_farthest_artist([])
                    plt.pause(0.001)
            else:
                theta0 = np.arctan2(a0_rel[1], a0_rel[0])
                thetas = np.linspace(theta0, theta0 + np.deg2rad(digree), steps) if steps != 1 else [theta0 + np.deg2rad(digree)]

                mapper = BetterFitwithGaussianRBF(
                    vertices=list(self.pos.values()),
                    src=self.src_ctrl,
                    K=K_threshold,
                    epsilon=70.0,
                    fps_k=fps_k
                )
                mapper.initialize_first_step()

                for t in thetas:
                    dst_ctrl = self.src_ctrl.copy()
                    dst_ctrl[0] = center + np.array([r * np.cos(t), r * np.sin(t)])
                    mapper.dst = dst_ctrl

                    mapper.update_active_set()
                    mapper.optimize_step()
                    mapper.postprocess()

                    # pos を更新
                    new_positions = {i: pt for i, pt in enumerate(mapper.Phi @ mapper.c.T)}
                    self.pos = new_positions

                    # activated と farthest を可視化（mapper 側の形式に合わせて安全に処理）
                    activated_indices = list(mapper.activated) if getattr(mapper, "activated", None) is not None else []
                    self._update_activated_artist(activated_indices)

                    farthest = getattr(mapper, "farthest_points", None)
                    # farthest が None か、配列か、boolean mask のどれでも対応
                    if farthest is None:
                        self._update_farthest_artist([])
                    else:
                        # numpy array / list / boolean mask をそのまま渡す（内部で処理）
                        self._update_farthest_artist(farthest)

                    # 制御点（dst）の可視化を更新
                    self._update_ctrl_artists(dst_ctrl)

                    self._update_artists()
                    plt.pause(0.01)

        finally:
            self._hide_status()
            self._set_buttons_visible(True)
            self.is_computing = False

    def _update_activated_artist(self, activated_indices: List[int]):
        """activated_indices の頂点位置に合わせて activated_scatter を更新する。
           activated_indices が空なら offsets を空にする。
        """
        if activated_indices is None or len(activated_indices) == 0:
            # 空にするには空配列を渡す
            self.activated_scatter.set_offsets(np.empty((0, 2)))
        else:
            # self.pos のキーは頂点インデックスである前提
            pts = np.array([self.pos[idx] for idx in activated_indices])
            self.activated_scatter.set_offsets(pts)
        # 描画を反映
        self.fig.canvas.draw_idle()

    def _update_artists(self):
        # ノード座標を更新（既存）
        offsets = np.array([self.pos[n] for n in self.node_list])
        self.nodes_pc.set_offsets(offsets)

        # エッジを更新
        segments = [(self.pos[u], self.pos[v]) for u, v in self.G.edges()]
        try:
            self.edge_lc.set_segments(segments)
        except Exception:
            # 既存のコレクション周りの例外対処
            self.ax.collections = [c for c in self.ax.collections if c is self.nodes_pc]
            nx.draw_networkx_edges(self.G, pos=self.pos, ax=self.ax)

        # ラベル位置を更新
        for n, text in self.labels_dict.items():
            text.set_position(self.pos[n])

        offsets = np.array([self.pos[n] for n in self.node_list])
        self.nodes_pc.set_offsets(offsets)

        segments = [(self.pos[u], self.pos[v]) for u, v in self.G.edges()]
        try:
            self.edge_lc.set_segments(segments)
        except Exception:
            self.ax.collections = [c for c in self.ax.collections if c is self.nodes_pc]
            nx.draw_networkx_edges(self.G, pos=self.pos, ax=self.ax)

        for n, text in self.labels_dict.items():
            text.set_position(self.pos[n])

        # ---------------------------
        # ここから表示領域を自動調整して全体を常に表示する処理
        # ---------------------------
        if offsets.size == 0:
            return  # 安全策

        xs = offsets[:, 0]
        ys = offsets[:, 1]
        # さらに制御点も表示領域に含めるようにする
        all_x = np.concatenate([xs, self.src_ctrl[:, 0], self.current_dst_ctrl[:, 0]])
        all_y = np.concatenate([ys, self.src_ctrl[:, 1], self.current_dst_ctrl[:, 1]])

        minx, maxx = all_x.min(), all_x.max()
        miny, maxy = all_y.min(), all_y.max()

        width = maxx - minx
        height = maxy - miny

        # 極端に小さい場合に見やすい最小スパンを確保
        min_span = 1.0  # 必要に応じて調整
        width = max(width, min_span)
        height = max(height, min_span)

        # 等倍率を保つため、幅の大きい方に合わせて表示範囲を決める
        half_span = max(width, height) / 2.0

        # 少し余白を取る（10%）
        pad = 0.1 * (2 * half_span)
        half_span = half_span + pad / 2.0

        center_x = (maxx + minx) / 2.0
        center_y = (maxy + miny) / 2.0

        self.ax.set_xlim(center_x - half_span, center_x + half_span)
        self.ax.set_ylim(center_y - half_span, center_y + half_span)

        # 等倍を再設定（念のため）
        self.ax.set_aspect('equal', adjustable='box')

        # 描画更新
        self.fig.canvas.draw_idle()


    def update(self, val=None):
        # self.pos[self.a_key] = self.state['a_pos']
        self._update_artists()

    def reset_positions(self):
        self.pos.update(self.initial_pos)
        self.state['a_pos'] = self.initial_pos[self.a_key]
        # dst を src と同じに戻す
        self._update_ctrl_artists(self.src_ctrl.copy())
        # activated と farthest を空に戻す
        self._update_activated_artist([])
        self._update_farthest_artist([])
        self.update(None)


if __name__ == "__main__":
    # カレントディレクトリ基準でファイルパス構築
    base = os.getcwd()
    fn_points = os.path.join(base, "create_mesh\\intern-2025\\sample006.csv")
    fn_mesh = os.path.join(base, "create_mesh\\intern-2025\\points_mesh_index1.csv")

    # ----- データ準備 -----
    xs = np.linspace(1.0, 40.0, 20)
    ys = np.linspace(8.5, 13.5, 3)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel()]).T.astype(float)

    # 頂点キーとして整数またはインデックスを使う
    pos_dict = {i: pt for i, pt in enumerate(pts)}

    # Graph 作成と三角形接続
    G = nx.Graph()
    G.add_nodes_from(pos_dict.keys())
    tris= Delaunay(pts)
    for tri in tris.simplices:  # tri は3つのインデックス
        u, v, w = tri
        for a, b in [(u, v), (v, w), (w, u)]:
            G.add_edge(a, b)

    # 'a' に相当するノードキーを定義。例えば、0番目とする。
    a_key = 0

    # 任意の src_ctrl を渡したいならここで指定できます（例）
    custom_src_ctrl = np.array([[1.0, 8.0], [33.0, 13.0], [40.0, 9.0], [40.0, 13.0]])

    ui = ProgressiveAnimationGraphUI(G=G, initial_pos=pos_dict, a_key=a_key,
                                     src_ctrl=custom_src_ctrl, figsize=(8,6))
    ui.start()
