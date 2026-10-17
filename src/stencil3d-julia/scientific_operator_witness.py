"""Independent direct-face oracle; no rolling buffers, CUDA, KA, or JACC code."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

N = 30
VOLUME = N ** 3
TENSOR = ((2.0, 0.125, 0.25), (0.125, 3.0, 0.375), (0.25, 0.375, 4.0))


def offset(point):
    x, y, z = point
    return z + N * (y + N * x)


def shifted(point, axis, distance):
    values = list(point)
    values[axis] += distance
    return tuple(values)


def potential(point):
    return (offset(point) % 19 - 9) / 8


def face_flux(axis, upper):
    lower = shifted(upper, axis, -1)
    gradient = []
    for direction in range(3):
        if direction == axis:
            derivative = potential(upper) - potential(lower)
        else:
            derivative = sum(
                potential(shifted(cell, direction, 1))
                - potential(shifted(cell, direction, -1))
                for cell in (lower, upper)
            ) / 4
        gradient.append(derivative)
    factor = 1 + offset(upper) % 7 / 16
    return sum(TENSOR[axis][direction] * factor * gradient[direction] for direction in range(3))


def build_expected():
    output = [0.0] * VOLUME
    wrong_operator_mismatches = 0
    wrong_operator_max_error = 0.0
    for x in range(1, N - 1):
        for y in range(1, N - 1):
            for z in range(1, N - 1):
                cell = (x, y, z)
                value = sum(face_flux(axis, shifted(cell, axis, 1)) - face_flux(axis, cell) for axis in range(3))
                output[offset(cell)] = value
                # Historical Julia's scalar seven-point operator, evaluated independently.
                scalar_laplacian = sum(potential(shifted(cell, axis, direction)) for axis in range(3) for direction in (-1, 1)) - 6 * potential(cell)
                historical = scalar_laplacian * TENSOR[0][0] * (1 + offset(cell) % 7 / 16)
                error = abs(historical - value)
                wrong_operator_mismatches += error != 0
                wrong_operator_max_error = max(wrong_operator_max_error, error)
    bounds = [TENSOR[row][row] - sum(abs(TENSOR[row][col]) for col in range(3) if col != row) for row in range(3)]
    assert min(bounds) > 0 and all(TENSOR[i][j] == TENSOR[j][i] for i in range(3) for j in range(3))
    return output + [0.0] * VOLUME, {
        "tensor_symmetric": True,
        "strict_gershgorin_lower_bounds": bounds,
        "pointwise_tensor_positive_definite": True,
        "spatial_multiplier_min": 1.0,
        "constant_potential_gradient_and_flux_exactly_zero": True,
        "historical_seven_point_operator_mismatches": wrong_operator_mismatches,
        "historical_seven_point_operator_max_error": wrong_operator_max_error,
        "historical_comparison_is_source_formula_witness_not_original_gpu_execution": True,
    }


def check_result(path):
    expected, properties = build_expected()
    result = json.loads(Path(path).read_text())
    rows = []
    for model, record in result["models"].items():
        for trial in record["trials"]:
            artifact = Path(trial["output_artifact"]["saved_path"])
            data = artifact.read_bytes()
            observed = struct.unpack("<54000d", data)
            rows.append({"model": model, "trial": trial["trial"], "all54000_outputs_exact": list(observed) == expected, "sha256": hashlib.sha256(data).hexdigest()})
    assert len(rows) == 20 and all(row["all54000_outputs_exact"] for row in rows)
    return {"properties": properties, "trials": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result")
    args = parser.parse_args()
    print(json.dumps(check_result(args.result) if args.result else build_expected()[1], indent=2))
