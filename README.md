# MOSS-Image LoRA Trainer GUI for Windows

MOSS-ImageのLoRAをWindows上で作成するためのブラウザ操作型GUIです。画像・キャプションの確認、WD14タグ生成、タグ編集、学習量の計算、学習の開始・停止、ログ確認までを一画面で行えます。

> Windows 11・GeForce RTX 4070 12GBで動作確認済みです。現時点ではWindows＋NVIDIA GPU専用です。

## 主な機能

- 画像と同名の`.txt`キャプションをサムネイル付きで表示
- 画像一覧を9／12／15枚単位でページ切り替え
- WD14 Taggerによる不足キャプションの自動生成
- タグのクリック削除、個別追加・保存
- 固有トリガーワードの一括追加、指定タグの一括削除
- 画像枚数に応じた学習強度候補と予定step数の表示
- ConvRot INT8を利用した12GB VRAM向け学習
- 学習工程、step進捗、直近ログのブラウザ表示
- 任意のLANアクセス

モデル、学習画像、生成LoRA、Slack連携機能は含まれません。

## 動作要件

- Windows 10／11
- NVIDIA GPU（VRAM 12GBを基準、RTX 4070 12GBで検証）
- NVIDIAドライバー
- [Python 3.10（64-bit）](https://www.python.org/downloads/release/python-31011/)
  - インストール時に`Add Python to PATH`とPython Launcherを有効にしてください。
- [Git for Windows](https://git-scm.com/download/win)
- RAM、ページファイル、モデル・キャッシュ用の十分な空き容量

VRAM 8GB以下、AMD／Intel GPU、Linux、macOSは未検証・サポート対象外です。

## 必要なモデル

モデルは同梱されません。次の3ファイルを用意し、`config.json`に絶対パスを設定します。

| 用途 | 必要なファイル例 | 注意 |
|---|---|---|
| MOSS本体 | `moss_image_base_int8_convrot.safetensors` | **SFT版ではなくBase版** |
| Text Encoder | `qwen3.5_9b_int8_convrot.safetensors` | W4A8 GPTQ版ではなくConvRot INT8版 |
| VAE | `flux2-vae_small.safetensors`など | 実際のファイル名を設定 |

- 公式モデル: [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)
- 学習実装と詳細: [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)

単一ファイルの量子化モデルは公式とは別の配布元にある場合があります。ダウンロード前に配布元、ファイル名、ライセンス、ハッシュを確認してください。

## インストール

1. このリポジトリをcloneするか、`Code > Download ZIP`から展開します。
2. `setup_windows.cmd`を実行します。
3. 作成された`config.json`を開き、モデル3点と出力先を設定します。
4. `start_gui.cmd`を実行します。
5. `http://127.0.0.1:8004`を開きます。

初回はPyTorchなどのパッケージと`sd-scripts`をダウンロードするため時間がかかります。ComfyUIのPython環境は変更せず、このツール専用の`.venv`を使います。

### `config.json`の例

Windowsのパスはスラッシュで書くとJSONのエスケープミスを避けられます。

```json
{
  "listen_host": "127.0.0.1",
  "port": 8004,
  "python": ".venv/Scripts/python.exe",
  "sd_scripts": "sd-scripts",
  "base_model": "D:/AI/models/moss_image_base_int8_convrot.safetensors",
  "text_encoder": "D:/AI/models/qwen3.5_9b_int8_convrot.safetensors",
  "vae": "D:/AI/models/flux2-vae_small.safetensors",
  "output_dir": "D:/AI/models/loras/MOSS-Image"
}
```

`config.json`は個人パスを含むためGit管理から除外されます。共有用の雛形は`config.example.json`です。

## 学習データ

画像と同じフォルダに同名の`.txt`を置きます。PNG、JPG、JPEG、WEBPに対応します。

```text
training-data/
├─ 001.png
├─ 001.txt
├─ 002.jpg
└─ 002.txt
```

`.txt`がなければGUIからWD14タグ生成を実行できます。生成後は内容を確認し、学習させたくない要素を削除してください。

## 基本操作

1. 学習画像フォルダを入力して一覧を読み込みます。
2. 不足する`.txt`へWD14タグを生成します。
3. トリガーワードを入力し、一括追加ボタンを押します。
4. グリッド上でキャプションを確認・編集します。
5. 出力名、解像度、学習強度を確認します。
6. 右上の「学習開始」を押し、最下部の進捗を確認します。

トリガーワードは入力しただけでは保存されません。キャプションが空の画像が残っている間は学習を開始できません。

## 学習強度

予定step数は `画像枚数 × 固定10回 × 学習強度` です。

| 学習強度 | 目安 |
|---|---|
| 軽め（5） | 40枚以上 |
| 標準（10） | 15～39枚 |
| 強め（20） | 10枚以下、または特徴が弱い場合 |

これは開始点の目安です。過学習が見られる場合は途中保存されたLoRAまたは低い学習強度を試してください。

## LAN内の別PCから操作する

`config.json`の`listen_host`を`0.0.0.0`へ変更し、Windowsファイアウォールで使用ポートを許可します。別PCから`http://学習PCのIP:8004`を開きます。

このGUIには認証がありません。インターネットへ公開せず、信頼できるLAN内だけで使用してください。

## 出力とローカルデータ

- LoRA: `config.json`の`output_dir`
- 実行ログと状態: `runtime/`
- 自動生成した学習設定: `configs/`
- WD14モデル: Hugging Faceの標準キャッシュ

`runtime/`、`configs/`、`config.json`はGit管理から除外されます。

## 更新

更新後、依存関係や`sd-scripts`の固定リビジョンが変わった場合は`setup_windows.cmd`を再実行してください。既存の`config.json`は上書きされません。

## トラブルシューティング

[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)を参照してください。不具合報告ではGPU、VRAM、Windows版、Python版、再現手順、画面のエラー、ログ末尾を添えてください。ログ中の個人パスは伏せてください。

## セキュリティとライセンス

- セキュリティ方針: [SECURITY.md](SECURITY.md)
- 学習処理: [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- 自動タグ生成: [SmilingWolf WD14 Tagger](https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2)
- MOSS-Image: [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)

このリポジトリのコードはMIT Licenseです。外部モデル、学習スクリプト、Pythonパッケージにはそれぞれのライセンスが適用されます。
