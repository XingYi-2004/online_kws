"""构建 1000 条回放库：357 干净测试（全 29 类）+ data_train 训练集分层抽样 643 条。

抽样规则：每类 22 条，类内优先"真车（car）样本"和"低 SNR simu（6~8dB）"，
不足 22 时用其余样本补齐；剩余 5 条分给样本最多的 5 个类。

用法：
  cd AlphaSNN && python alpha/asr/kws/tools/build_replay.py
输出：
  data/replay/replay_1000.jsonl.gz   （回放库清单）
"""
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from lhotse import CutSet, load_manifest_lazy


def category(cut) -> str:
    src = cut.recording.sources[0].source
    base = src.split("/")[-1]
    m = re.search(r"addsimu_(\d+)db", base)
    if "car" in base.lower():
        return "car"
    if m and int(m.group(1)) <= 8:
        return f"low_snr_{m.group(1)}db"
    if m:
        return "simu_high"
    return "other"


def main():
    random.seed(42)
    os_chdir = ROOT

    # ---- 357 干净测试 ----
    clean = list(CutSet.from_file(str(ROOT / "data/fbank/command_cuts_test_29.jsonl")))
    clean_ids = {c.id for c in clean}
    print(f"干净测试: {len(clean)} 条，类别覆盖 {len({int(c.supervisions[0].speaker) for c in clean})}/29")

    # ---- 训练集分层抽样 ----
    train_manifest = ROOT / "data_train/fbank/command_cuts_train_29.jsonl"
    cuts = load_manifest_lazy(str(train_manifest))
    by_class = defaultdict(list)
    for c in cuts:
        by_class[int(c.supervisions[0].speaker)].append(c)
    print(f"训练集: 类别覆盖 {len(by_class)}/29，各类样本数 min={min(len(v) for v in by_class.values())} "
          f"max={max(len(v) for v in by_class.values())}")

    overlap = sum(1 for c in (x for v in by_class.values() for x in v[:1]) if 0)  # placeholder
    # 检查干净测试 id 与训练集是否重叠（抽样验证 1 万条）
    overlap_ids = 0
    checked = 0
    for cls, pool in by_class.items():
        for c in pool:
            checked += 1
            if c.id in clean_ids:
                overlap_ids += 1
            if checked >= 50000:
                break
        if checked >= 50000:
            break
    print(f"训练集前 5 万条与干净测试 id 重叠: {overlap_ids}（应接近 0）")

    selected = []
    report = Counter()
    n_big = sorted(by_class.keys(), key=lambda k: -len(by_class[k]))[:5]  # 多分的 5 个类
    for cls in range(29):
        pool = by_class[cls]
        per = 23 if cls in n_big else 22
        car, low, other = [], [], []
        for c in pool:
            cat = category(c)
            if cat == "car":
                car.append(c)
            elif cat.startswith("low_snr"):
                low.append(c)
            else:
                other.append(c)
        random.shuffle(car)
        random.shuffle(low)
        random.shuffle(other)
        take = car[:11] + low[:11]
        need = per - len(take)
        if need > 0:
            take += other[:need]
        assert len(take) == per, f"class {cls}: 只有 {len(take)} 条可用"
        selected.extend(take)

    print(f"训练集抽样: {len(selected)} 条（期望 643）")
    print("抽样构成:")
    for k, v in Counter(category(c) for c in selected).most_common():
        print(f"  {v:5d}  {k}")

    combined = clean + selected
    out_dir = ROOT / "data/replay"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "replay_1000.jsonl.gz"
    CutSet.from_cuts(combined).to_file(str(out_path))
    print(f"已保存: {out_path}")

    # ---- 验证 ----
    reloaded = load_manifest_lazy(str(out_path))
    n = sum(1 for _ in reloaded)
    print(f"回读验证: {n} 条（期望 1000）")
    # 抽查特征可读
    for i, c in enumerate(list(reloaded)[:3]):
        f = c.load_features()
        print(f"  抽查 {i}: id={c.id[:40]}... feat={f.shape} label={int(c.supervisions[0].speaker)}")


if __name__ == "__main__":
    main()
