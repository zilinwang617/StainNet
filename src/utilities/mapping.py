from typing import Dict, List


# =========================================================
# 1. 真实 channel 顺序（按给定的 1~59 顺序）
#    Python 里 index 从 0 开始
# =========================================================
CHANNEL_NAMES: List[str] = [
    "ARF",  # 0
    "Akt",  # 1
    "BP1",  # 2
    "Bcl2",  # 3
    "CDK2",  # 4
    "CDK4",  # 5
    "CDK6",  # 6
    "Cdh1",  # 7
    "Cdt1",  # 8
    "D1",  # 9
    "DNA_1",  # 10
    "DNA_2",  # 11
    "DNA_3",  # 12
    "DNA_4",  # 13
    "DNA_5",  # 14
    "DNA_6",  # 15
    "DNA_7",  # 16
    "DNA_8",  # 17
    "E2F1",  # 18
    "ERK",  # 19
    "Fra1",  # 20
    "GSK3b",  # 21
    "PCNA-mTq2",  # 22
    "RSK1",  # 23
    "Rb",  # 24
    "S6",  # 25
    "STAT3",  # 26
    "Skp2",  # 27
    "YAP",  # 28
    "bCat",  # 29
    "cFos",  # 30
    "cJun",  # 31
    "cMyc",  # 32
    "cycA",  # 33
    "cycB1",  # 34
    "cycE",  # 35
    "p16",  # 36
    "p21",  # 37
    "p27",  # 38
    "p38",  # 39
    "p53",  # 40
    "pAkt",  # 41
    "pChk1",  # 42
    "pERK",  # 43
    "pGSK3b",  # 44
    "pH2AX",  # 45
    "pRSK",  # 46
    "pRb",  # 47
    "pS6",  # 48
    "pSTAT3",  # 49
    "pp21",  # 50
    "pp27",  # 51
    "pp38",  # 52
    "pp53",  # 53
    "pp65",  # 54
    "nuc",  # 55
    "cell",  # 56
    "cyto",  # 57
    "ring",  # 58
]


# =========================================================
# 2. baseline: 固定 6 input -> 1 target
# =========================================================
INPUT_INDICES: List[int] = [22, 24, 33, 34, 35, 47, 10]
TARGET_INDEX: int = 1


# =========================================================
# 3. 自动映射
# =========================================================
NAME_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CHANNEL_NAMES)}
IDX_TO_NAME: Dict[int, str] = {i: name for i, name in enumerate(CHANNEL_NAMES)}


# =========================================================
# 4. 安全检查
# =========================================================
def validate_mapping() -> None:
    assert len(CHANNEL_NAMES) == 59, f"期望 59 个通道，当前是 {len(CHANNEL_NAMES)}"
    assert len(CHANNEL_NAMES) == len(set(CHANNEL_NAMES)), "CHANNEL_NAMES 里有重复名字"
    assert len(INPUT_INDICES) == 7, f"baseline 要 7 个 input，但你现在给了 {len(INPUT_INDICES)} 个"
    assert TARGET_INDEX not in INPUT_INDICES, "TARGET_INDEX 不能和 input 重复"

    n = len(CHANNEL_NAMES)
    for idx in INPUT_INDICES:
        assert 0 <= idx < n, f"INPUT index 越界: {idx}"
    assert 0 <= TARGET_INDEX < n, f"TARGET index 越界: {TARGET_INDEX}"


# =========================================================
# 5. 常用函数
# =========================================================
def get_channel_names() -> List[str]:
    return CHANNEL_NAMES


def get_num_channels() -> int:
    return len(CHANNEL_NAMES)


def get_input_indices() -> List[int]:
    return INPUT_INDICES


def get_target_index() -> int:
    return TARGET_INDEX


def get_input_names() -> List[str]:
    return [IDX_TO_NAME[i] for i in INPUT_INDICES]


def get_target_name() -> str:
    return IDX_TO_NAME[TARGET_INDEX]


def name_to_index(name: str) -> int:
    return NAME_TO_IDX[name]


def index_to_name(index: int) -> str:
    return IDX_TO_NAME[index]


def names_to_indices(names: List[str]) -> List[int]:
    return [NAME_TO_IDX[name] for name in names]


def set_task_by_names(input_names: List[str], target_name: str) -> None:
    global INPUT_INDICES, TARGET_INDEX
    INPUT_INDICES = names_to_indices(input_names)
    TARGET_INDEX = name_to_index(target_name)
    validate_mapping()


# =========================================================
# 6. 如果直接运行，就打印当前任务设置
# =========================================================
if __name__ == "__main__":
    validate_mapping()
    print("=== Channel Mapping ===")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"{i:02d}: {name}")

    print("\n=== Current Task ===")
    print("Input indices:", INPUT_INDICES)
    print("Input names:  ", get_input_names())
    print("Target index: ", TARGET_INDEX)
    print("Target name:  ", get_target_name())
