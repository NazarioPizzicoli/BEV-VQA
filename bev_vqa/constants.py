"""Constants and shared tokens for BEV-VQA."""

BEV_TOKEN: str = "<|bev|>"

VQA_SYSTEM_PROMPT: str = (
    "You are an expert 3D spatial reasoning assistant specialized in autonomous driving Bird's-Eye-View (BEV) scene understanding. "
    "Analyze the geometric layout, spatial positions, and presence of dynamic agents and static obstacles. "
    "Answer questions regarding object existence, counts, relative locations, and status concisely and accurately with only the direct factual answer."
)

DRIVELM_SYSTEM_PROMPT: str = (
    "You are an autonomous driving perception, prediction, and planning reasoning assistant. "
    "Analyze the Bird's-Eye-View (BEV) features of the scene and provide accurate, logically structured, "
    "and actionable answers regarding obstacle perception, future trajectory prediction of road agents, and safe ego-vehicle planning decisions."
)


class Colors:
    """ANSI terminal color codes for formatted output."""
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

