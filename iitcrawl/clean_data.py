from glob import glob
from tqdm import tqdm


for file in tqdm(glob("./output/*.txt"), desc="Processing files", unit="file"):
    with open(file, "r") as f:
        lines = f.readlines()

    footer = False
    put = True
    prev = ""
    clean_lines = []

    for line in lines:
        line = line.strip()

        if line == "Home":
            put = False
            continue
        elif put:
            continue
        elif line == "" and prev == "":
            continue
        elif line == "__":
            put = True
        elif line.startswith("### Related"):
            put = True
        else:
            clean_lines.append(line.replace("_", ""))
        prev = line

    with open(file, "w") as f:
        f.write("\n".join(clean_lines))
