# GSviewer

2DGS / ArtiFixer3D / 3DGS の Gaussian PLY、RGB 点群 PLY、Open3D の面付き PLY と、
RGB 点群 PLY の末尾に埋め込まれた学習カメラ視点・カメラ軌跡を表示する、依存パッケージ
不要のローカルビューアーです。

指定データ `point_cloud_iteration_100000_with_train_views.ply` では、次を自動認識します。

- 3DGS 由来のシーン: 3,217,255 点
- 学習カメラ: 271 視点 × 6 方向
- カメラフラスタム: 208,128 点
- カメラ軌跡: 2,160 点

## すぐに使う

必要なものは Python 3.10 以降と、WebGL2 を有効にした Chrome / Safari / Firefox だけです。`pip install` は不要です。

初めて取得する場合:

```bash
git clone git@github.com:nishizawahiroki-oss/GSviewer.git
cd GSviewer
```

この Mac では対象 PLY が既定パスにあるため、次のコマンドだけで起動できます。

```bash
cd /Users/hiroki/Documents/3DGS-viewer
python3 view_3dgs.py
```

ブラウザが自動で `http://127.0.0.1:8000/` を開きます。終了するときは、ターミナルで `Ctrl+C` を押してください。

別の PLY を開く場合は、パスを指定します。

```bash
python3 view_3dgs.py /path/to/point_cloud.ply
```

Linux でも同じコマンドです。Windows PowerShell では次のように実行できます。

```powershell
py view_3dgs.py "C:\path\to\point_cloud.ply"
```

起動後に画面右上の「別の PLY」から選ぶこともできます。その場合、ファイルはブラウザ内で処理され、外部へアップロードされません。

## 光源パッケージを開く

`gaussian_without_light_sources.ply` と `light_sources.json` を同じディレクトリへ
SCP した場合は、その **ディレクトリのパス**を指定して起動します。

```bash
python3 view_3dgs.py \
  ~/trc-hires-12pinhole-sam3d-vlm-light-package-20260721
```

サーバーは次の構成を自動検出し、従来形式では PLY の頂点数と JSON の登録数が一致することを
起動前に検証します。

```text
trc-hires-12pinhole-sam3d-vlm-light-package-20260721/
├── gaussian_without_light_sources.ply
└── light_sources.json
```

Gaussian レンダラーの `Lights` パネルでは次をリアルタイムに変更できます。

- 全光源の ON / OFF と全体強度
- 120 灯から操作対象を選択
- レンダリング上の光源をクリックして操作対象を直接選択
- 灯具ごとの ON / OFF、強度、発光色

`light_sources.json` の位置・回転・寸法から 296 個の emissive Gaussian proxy を
ブラウザ内で生成し、背景の Gaussian と同じ深度ソートへ統合します。変更時は
光源 texel だけを更新するため、1.3 GB の PLY を再読み込みしません。

この JSON は `prototype_visual_proxy_only` で、各灯具にも
`affects_scene_surface_lighting: false` が指定されています。そのため現状の操作は
光源自体の発光表示に適用され、背景 3DGS 表面の物理的な再照明、影、間接光は
生成しません。これらには relightable appearance、法線・材質、または別の
relighting model/backend が必要です。

## メッシュ＋有限面光源ペアを開く

`mesh_ours_2pivots_post.ply` と、`emitters` 配列を持つ `light_sources.json` の
ペアにも対応しています。今回のようなディレクトリは、そのまま指定できます。

```bash
python3 view_3dgs.py \
  /Users/hiroki-airoa/trc-personmask-b4-30k-authoritative-light-mesh-pair-v1-20260723 \
  --port 9080
```

この形式の面付き PLY は、起動時に頂点と面の構造を検証し、現在の点群ビューアーで
メッシュ頂点を表示します。有限面光源の `emitter_count` と `sample_count`、無効化された
光源も含むJSONの整合性を起動前に検証します。表面の再照明・影生成は行いません。

## 操作方法

| 操作 | マウス / トラックパッド |
|---|---|
| 回転 | 左ドラッグ |
| 平行移動 | 右ドラッグ、または `Shift` + 左ドラッグ |
| ズーム | ホイール / 2 本指スクロール |
| 現在のフォーカスへ戻す | ビューポートをダブルクリック |
| 3DGS 全体へ移動 | 「シーンに合わせる」 |
| カメラ群へ移動 | 「カメラに合わせる」 |

右側のパネルでは、次を変更できます。

- 3DGS、学習カメラ、軌跡の個別表示
- 3DGS とカメラの点サイズ
- カメラを深度に関係なく前面表示する X-ray 表示（初期状態は ON）
- 背景色
- 271 視点の全表示、または 1 視点だけの表示

PLY に COLMAP の `camera` element（回転行列 + fx/fy/cx/cy + 画像サイズ）が
含まれる場合、学習視点は点サンプルではなく、内部パラメータから再構成した
正確なフラスタムを実線ワイヤーフレームで描画します（面 ID があれば 6 色で
色分け、なければ白）。`camera` element がない従来の PLY は点サンプル描画の
ままです。

カメラ色は PLY の定義どおりです。

| 方向 | 色 |
|---|---|
| Front | 青 |
| Right | 赤 |
| Back | 緑 |
| Left | 黄 |
| Up | 紫 |
| Down | シアン |

## Mac GPU / NVIDIA (CUDA 対応 GPU)

描画は WebGL2 で GPU アクセラレーションされます。画面左下に、ブラウザが公開している GPU renderer 情報が表示されます。プライバシー設定によっては、具体的な GPU 名ではなく一般名になることがあります。

- Apple Silicon / Intel Mac: Safari や Chrome の WebGL2 バックエンドを通して Mac GPU を使用
- Linux / Windows の NVIDIA 環境: NVIDIA GPU の WebGL2 / OpenGL / ANGLE バックエンドを使用
- CUDA Toolkit や PyTorch のインストール: 不要

NVIDIA の CUDA 対応 GPU を搭載したマシンでも GPU 描画できますが、このビューアーは CUDA compute kernel を直接呼ぶ実装ではありません。ブラウザのグラフィックス API を使うため、Mac と NVIDIA の両方を同じコードで扱えます。

## この PLY で可能な描画

対象 PLY の頂点属性は次の 6 個だけです。

```text
float x, y, z
uchar red, green, blue
```

元の Gaussian にあったはずの `scale`、`rotation`、`opacity`、高次 SH 係数は保存されていません。そのため、異方性楕円・半透明合成・視点依存色を使う「本来の Gaussian rasterization」は復元できません。このビューアーでは、保存されている Gaussian 中心を色付きの丸いスクリーン空間スプラットとして描きます。

完全な 3DGS PLY（scale / rotation / opacity / SH を含む）に対しては、次節の Gaussian レンダラー経路が使えます。

## 2DGS / ArtiFixer3D / 3DGS Gaussian レンダラー (`/gs`)

2DGS PLY（`scale_0..1`）と完全な 3DGS PLY（`scale_0..2`）を自動判別します。
`artifixer3d.ply` はファイル名から ArtiFixer3D として識別します。
どちらも `f_dc_0..2` / `opacity` / `rot_0..3` を持つ Gaussian PLY を渡すと、
ブラウザは `http://127.0.0.1:8000/gs` の Gaussian レンダラーで開きます。

```bash
python3 view_3dgs.py /path/to/point_cloud.ply
```

`method_ply_30k` の比較データは次のように個別に開けます。

```bash
python3 view_3dgs.py /Users/hiroki-airoa/method_ply_30k/2dgs.ply
python3 view_3dgs.py /Users/hiroki-airoa/method_ply_30k/artifixer3d.ply
```

描画内容:

- 2DGS: 公式 `diff-surfel-rasterization` の tangent-plane homography、AABB、
  pixel-ray / surfel 交差（論文の式 8–10）、low-pass filter を WebGL2 に移植
- ArtiFixer3D: 公式 `3DGRUT-ArtiFixer` の 7 sigma-point Unscented Transform、
  covariance dilation / Mip opacity compensation、Gaussian-local ray density を WebGL2 に移植
- 3DGS: 3D 共分散 Σ = R S² Rᵀ を EWA スプラッティングで射影
- sigmoid(opacity) を使った front-to-back の半透明合成（深度は worker のカウンティングソート）
- SH DC 項からの色（視点依存の高次 SH 係数は未適用）
- `filter_3D` プロパティがある場合は Mip-Splatting の 3D フィルタ補正を適用

移植元は 2DGS の pinned submodule
[`hbb1/diff-surfel-rasterization@e0ed020`](https://github.com/hbb1/diff-surfel-rasterization/tree/e0ed0207b3e0669960cfad70852200a4a5847f61)
と、ArtiFixer の pinned submodule
[`nv-tlabs/3DGRUT-ArtiFixer@62e1038`](https://github.com/nv-tlabs/3DGRUT-ArtiFixer/tree/62e1038b74b2edc01440fd4ddf5f080109b6faba)
です。CUDA / OptiX backend 自体をブラウザで実行するのではなく、前向きレンダリングの
投影・support・density の式を GLSL に移しています。出典とライセンスは
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) を参照してください。
- 1.4 GB / 500 万ガウシアン級を WebGL2 テクスチャ 1 枚 + インスタンス描画で表示

右パネルで次を調整できます。

- スプラットスケール（楕円の倍率。0.05× にすると点群相当の表示になり形状確認に便利）
- 不透明度しきい値（低 opacity のフローターを間引く。動画学習由来のもや対策）
- 軸反転 X / Y / Z（COLMAP 系の座標で上下・裏表が逆に見えるときのミラー表示。
  位置と共分散楕円の両方に適用されます）
- 背景色
- 光源パッケージ時の emissive proxy の全体・個別操作

画面左下の軸ギズモに、データの +X（赤）+Y（緑）+Z（青）が画面上で
どちらを向いているかが常時表示されます。軸反転の状態も反映されます
（視点向こう側を向く軸は薄く表示）。

注意: 動画から学習した 3DGS は、学習カメラ軌跡から離れた視点では
巨大な半透明ガウシアンにより霧がかかったように見えます。これはデータ由来で、
学習視点付近（シーン内部）では正しく写実的に描画されます。

パーサーは頂点 stride をヘッダーから決めるストリーミング実装のため、
`nx/ny/nz` や `f_rest_*`、`gaussian_features_*` など追加の float プロパティが
混在していても読めます。`red/green/blue`(uchar) しか持たない PLY を `/gs` で
開いた場合は、その色を使って等方点として描画します。

## カメラの自動分離

この PLY では、カメラは独立した `camera` element ではなく、頂点配列の末尾に色付き点として追加されています。ビューアーは末尾から連続ブロックを検証し、次の範囲を分離します。

```text
scene → back → down → front → left → right → up → white trajectory
```

シーン側にも完全な白点が 21,927 点あるため、RGB が白かどうかだけでは軌跡を判定していません。末尾の連続ブロック位置、6 色の順序、各ブロックの同一長、`271 × 128` 点、`270 × 8` 軌跡点が整合することを確認します。

カメラブロックがない通常の RGB PLY も、カメラなしの点群として表示できます。対応形式は `binary_little_endian PLY 1.0` です。現状は vertex element のみを持つ PLY が対象で、点以外の face / edge element にデータが入ったメッシュ PLY は対象外です。

## コマンドオプション

PLY を解析だけして、ブラウザを開かない場合:

```bash
python3 view_3dgs.py /path/to/file.ply --inspect-only
```

JSON メタデータを表示する場合:

```bash
python3 view_3dgs.py /path/to/file.ply --inspect-only --json
```

ポートを変更する場合:

```bash
python3 view_3dgs.py /path/to/file.ply --port 8080
```

ブラウザを自動で開かない場合:

```bash
python3 view_3dgs.py /path/to/file.ply --no-browser
```

同じ LAN の別端末からアクセスさせる場合:

```bash
python3 view_3dgs.py /path/to/file.ply --host 0.0.0.0
```

この場合は PLY が LAN 内から取得可能になるため、信頼できるネットワークだけで使用してください。

全オプション:

```bash
python3 view_3dgs.py --help
```

## テスト

追加依存なしで実行できます。

```bash
python3 -m unittest discover -s tests -v
```

実データの検証:

```bash
python3 view_3dgs.py \
  /Users/hiroki/point_cloud_iteration_100000_with_train_views.ply \
  --inspect-only
```

## トラブルシュート

### `WebGL2 を利用できません`

Chrome / Safari / Firefox の最新版を使い、ブラウザ設定でハードウェアアクセラレーションを有効にしてください。リモートデスクトップや仮想マシンでは WebGL2 が無効になることがあります。

### NVIDIA GPU が表示されない

画面左下の GPU 名を確認してください。ノート PC のハイブリッド GPU では、OS または NVIDIA Control Panel でブラウザを高パフォーマンス GPU に割り当てる必要がある場合があります。

### 点群が小さく見える

このデータには少数の遠い外れ点があります。初期表示と「シーンに合わせる」は 1–99 パーセンタイル範囲を使い、外れ点による過度なズームアウトを避けています。カメラを詳しく見る場合は「カメラに合わせる」を使用してください。

### ポート 8000 が使用中

空きポートを自動選択できます。

```bash
python3 view_3dgs.py /path/to/file.ply --port 0
```

## 構成

```text
view_3dgs.py       PLY / 光源パッケージ検証とローカル HTTP サーバー
web/index.html     点群ビューアー UI
web/viewer.js      WebGL2 点描画と操作
web/ply-worker.js  バックグラウンド PLY 解析（点群用）
web/gs.html        Gaussian レンダラー UI (/gs)
web/gs-viewer.js   2DGS surfel / ArtiFixer 3DGUT / 3DGS EWA 描画
web/gs-worker.js   Gaussian PLY ストリーミング解析・方式判定・深度ソート
tests/             PLY 分離・HTTP 配信のテスト
THIRD_PARTY_NOTICES.md  公式実装の移植範囲・出典・ライセンス案内
licenses/          同梱した第三者ライセンス全文
```
