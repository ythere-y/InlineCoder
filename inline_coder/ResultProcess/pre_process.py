import os


def PreProcessDevEvaldata(data):
    for sample in data:
        if "target_file" not in sample:
            target_file_path = os.path.join(
                sample["target_project_path"],
                "/".join(sample["completion_path"].split("/")[2:]),
            )
            with open(target_file_path, "r", encoding="utf-8") as f:
                sample["target_file"] = f.read()
            # get target_signature
            file_lines = sample["target_file"].split("\n")
            signature_lines = file_lines[
                sample["signature_position"][0] - 1 : sample["signature_position"][1]
            ]
            body_lines = file_lines[
                sample["body_position"][0] - 1 : sample["body_position"][1]
            ]
            sample["target_body"] = "\n".join(body_lines)
            sample["target_signature"] = "\n".join(signature_lines)
    return data
