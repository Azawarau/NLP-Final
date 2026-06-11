#!/usr/bin/env python
"""终端验证：检查基础阶段代码、编码与（可选）MTEB 小规模试跑。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED_FILES = [
    "src/prompts.py",
    "src/pooling.py",
    "src/llm_encoder.py",
    "src/evaluate.py",
    "scripts/run_basic.py",
    "scripts/run_layer_ablation.py",
    "configs/basic.yaml",
    "reports/basic_experiment.md",
]

REQUIRED_CAPABILITIES = [
    ("PromptEOL 模板", "src/prompts.py", "one word"),
    ("mean-pooling", "src/pooling.py", "mean_pooling"),
    ("last-token (PromptEOL)", "src/pooling.py", "last_token_pooling"),
    ("层索引抽取", "src/llm_encoder.py", "_resolve_layer_index"),
    ("MTEB 三数据集任务", "src/evaluate.py", "LEMBQMSumRetrieval"),
]


def check_files() -> list[str]:
    errors = []
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).exists():
            errors.append(f"缺少文件: {rel}")
    return errors


def check_capabilities() -> list[str]:
    errors = []
    for name, rel, needle in REQUIRED_CAPABILITIES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        if needle not in text:
            errors.append(f"能力未实现: {name} ({rel})")
    return errors


def check_imports() -> list[str]:
    errors = []
    for mod in ("torch", "transformers", "numpy", "yaml"):
        try:
            __import__(mod)
        except ImportError:
            errors.append(f"未安装: {mod}")
    try:
        import mteb  # noqa: F401
    except ImportError:
        errors.append("未安装: mteb（完整评估需要）")
    return errors


def run_smoke() -> list[str]:
    import subprocess

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return [f"smoke_test 失败:\n{r.stdout}\n{r.stderr}"]
    return []


def run_offline_pipeline() -> list[str]:
    import subprocess

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_offline_encoder.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return [f"verify_offline_encoder 失败:\n{r.stdout}\n{r.stderr}"]
    return []


def run_encoder_probe(
    use_model: bool, fallback: bool, tiny: bool
) -> tuple[list[str], dict | None]:
    if not use_model:
        return [], None

    from src.llm_encoder import load_encoder

    if tiny:
        model_name = "sshleifer/tiny-gpt2"
    elif fallback:
        model_name = "Qwen/Qwen2-1.5B-Instruct"
    else:
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    info: dict = {"model": model_name, "methods": {}}
    texts = [
        "The capital of France is Paris.",
        "Machine learning enables computers to learn from data.",
    ]
    errors = []
    for method in ("mean", "prompteol"):
        for layer in (-1, 8):
            try:
                enc = load_encoder(
                    model_name=model_name,
                    method=method,
                    layer=layer,
                    max_length=256,
                    batch_size=1,
                )
                emb = enc.encode(texts)
                key = f"{method}_layer{layer}"
                info["methods"][key] = {"shape": list(emb.shape), "norm": float((emb**2).sum() ** 0.5)}
                del enc
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                errors.append(f"编码探测失败 {method} layer={layer}: {e}")
    return errors, info


def run_mteb_quick(fallback: bool) -> tuple[list[str], dict | None]:
    try:
        import mteb
    except ImportError:
        return ["跳过 MTEB 试跑（未安装 mteb）"], None

    from src.evaluate import run_mteb_evaluation
    from src.llm_encoder import load_encoder

    model_name = "Qwen/Qwen2-1.5B-Instruct" if fallback else "mistralai/Mistral-7B-Instruct-v0.3"
    out = ROOT / "results" / "verify_quick"
    enc = load_encoder(
        model_name=model_name,
        method="mean",
        layer=-1,
        max_length=512,
        batch_size=1,
    )
    # ArguAna 相对较小，用于快速验证 MTEB 管线
    try:
        metrics = run_mteb_evaluation(
            enc,
            tasks=["ArguAna"],
            output_dir=str(out),
            batch_size=1,
            overwrite=True,
        )
        return [], {"mteb_quick": metrics}
    except Exception as e:
        return [f"MTEB 试跑失败: {e}"], None
    finally:
        del enc


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--with-model", action="store_true", help="加载 HF 模型做编码探测")
    ap.add_argument("--with-mteb", action="store_true", help="在 ArguAna 上做一次完整 MTEB 试跑")
    ap.add_argument("--fallback", action="store_true", help="使用 Qwen2-1.5B")
    ap.add_argument("--mistral", action="store_true", help="使用 Mistral-7B（需大显存）")
    ap.add_argument(
        "--tiny",
        action="store_true",
        help="使用 sshleifer/tiny-gpt2 做离线编码探测（无需大显存/稳定下载）",
    )
    args = ap.parse_args()
    fallback = args.fallback or (not args.mistral and not args.tiny)

    all_errors: list[str] = []
    report: dict = {"checks": {}}

    for name, fn in [
        ("files", check_files),
        ("capabilities", check_capabilities),
        ("imports", check_imports),
        ("smoke", run_smoke),
        ("offline_pipeline", run_offline_pipeline),
    ]:
        errs = fn()
        report["checks"][name] = "PASS" if not errs else "FAIL"
        all_errors.extend(errs)

    if args.with_model:
        errs, probe = run_encoder_probe(True, fallback, args.tiny)
        report["encoder_probe"] = probe
        report["checks"]["encoder"] = "PASS" if not errs else "FAIL"
        all_errors.extend(errs)

    if args.with_mteb:
        errs, mteb_res = run_mteb_quick(fallback)
        report["mteb"] = mteb_res
        report["checks"]["mteb"] = "PASS" if not errs else "FAIL"
        all_errors.extend(errs)

    out_path = ROOT / "results" / "verify_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n========== 基础阶段验证报告 ==========\n")
    for k, v in report["checks"].items():
        print(f"  [{v}] {k}")
    if report.get("encoder_probe"):
        print(f"\n  编码探测: {json.dumps(report['encoder_probe'], ensure_ascii=False)}")
    if report.get("mteb"):
        print(f"\n  MTEB 试跑: {json.dumps(report['mteb'], ensure_ascii=False)}")

    # 对照 proposal 基础阶段条目
    proposal_items = [
        "Mistral-Instruct-7B-0.3 支持（代码默认，可用 --mistral 实测）",
        "PromptEOL 实现",
        "mean-pooling 实现",
        "不同 Transformer layer 比较（run_layer_ablation.py）",
        "QMSum / 2WikiMultihop / ArguAna 评估配置",
    ]
    code_complete = not any(
        e for e in all_errors if "缺少文件" in e or "能力未实现" in e or "smoke" in e
    )
    print("\n--- proposal 基础阶段对照 ---")
    for item in proposal_items:
        print(f"  · {item}")
    print(f"\n代码与单元自检: {'通过' if code_complete else '未通过'}")

    if all_errors:
        print("\n失败项:")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)

    if not args.with_model:
        print("\n提示: 加 --with-model 验证模型编码；加 --with-mteb 验证 MTEB 管线。")
    print(f"\n报告已保存: {out_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
