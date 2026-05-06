import argparse
import json

import torch

from .transfuserpp_bridge import load_transfuserpp


def _match(name: str, filters):
    if not filters:
        return True
    return any(token in name for token in filters)


def main() -> None:
    parser = argparse.ArgumentParser(description="List CARLA Garage TransFuser++ parameter names for selective fine-tuning.")
    parser.add_argument("--garage-root", required=True)
    parser.add_argument("--team-config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--filter", default="", help="Comma-separated substrings to limit the printed parameter list.")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    filters = [item.strip() for item in args.filter.split(",") if item.strip()]
    device = torch.device("cpu")
    net, _, load_info = load_transfuserpp(args.garage_root, args.team_config, checkpoint=args.checkpoint, device=device)

    rows = []
    total = 0
    matched = 0
    for name, param in net.named_parameters():
        total += 1
        if not _match(name, filters):
            continue
        matched += 1
        if len(rows) < args.limit:
            rows.append({"name": name, "shape": list(param.shape), "count": int(param.numel())})

    print(
        json.dumps(
            {
                "load_info": load_info,
                "filters": filters,
                "total_parameter_tensors": total,
                "matched_parameter_tensors": matched,
                "preview": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
