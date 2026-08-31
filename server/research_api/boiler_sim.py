"""Unity 面板用确定性锅炉热力演示模拟（非机理级）。"""

from __future__ import annotations

from typing import Any


def run_boiler_simulation(
    *,
    coal_feed: float,
    air_flow: float,
    water_flow: float,
    drum_pressure: float,
    slag_degree: float = 0.0,
) -> dict[str, Any]:
    """按 Unity 面板输入范围（给煤 60~180 t/h、送风 600~1800 km³/h、
    给水 400~1100 t/h、汽包压力 12~20 MPa、结渣 0~0.8）的线性演示公式。

    返回 steam_output(t/h)、wall_temp(°C)、state_code/state_name 与原始参数。
    """
    coal_feed = float(coal_feed)
    air_flow = float(air_flow)
    water_flow = float(water_flow)
    drum_pressure = float(drum_pressure)
    slag_degree = float(slag_degree)

    steam_output = (
        coal_feed * 4.2
        + air_flow * 0.22
        + water_flow * 0.62
        - drum_pressure * 12.0
        - slag_degree * 150.0
    )
    wall_temp = (
        400.0
        + coal_feed * 0.25
        + air_flow * 0.02
        - water_flow * 0.02
        + drum_pressure * 2.0
        - slag_degree * 40.0
    )
    state_code = 1 if (drum_pressure >= 20.5 or wall_temp >= 470.0) else 0
    state_name = "过热预警" if state_code == 1 else "正常"
    return {
        "steam_output": round(max(0.0, steam_output), 3),
        "wall_temp": round(wall_temp, 2),
        "state_code": state_code,
        "state_name": state_name,
        "parameters": {
            "coal_feed": coal_feed,
            "air_flow": air_flow,
            "water_flow": water_flow,
            "drum_pressure": drum_pressure,
            "slag_degree": slag_degree,
        },
    }
