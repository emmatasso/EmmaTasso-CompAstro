import argparse

def balltime(h, g=9.8):
    return (2*h/g)**0.5

parser = argparse.ArgumentParser(description="Compute fall time from height h.")
parser.add_argument("h", type=float, help="height (m)")
parser.add_argument("-g", "--gravity", type=float, default=9.8, help="gravity (m/s^2)")

args = parser.parse_args(args=["20", "-g", "9.8"])
print("time =", balltime(args.h, args.gravity), "s")
