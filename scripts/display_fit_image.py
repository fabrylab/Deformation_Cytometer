from pathlib import Path

#file:
path = Path(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_16_alginate2%_NIH_tanktreading\2")
output = Path("image.html")

with output.open("w") as fp:
    for file in path.glob("*\\fit*.png"):
        fp.write(f"<img src='file:{file}' />\n")
        print("file", f"<img src='file:{file}' />\n")
