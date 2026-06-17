"""End-to-end: generate dataset -> train + calibrate -> fairness audit + mitigation."""
import argparse, logging, os
from generate_data import generate
import train, fairness

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    os.makedirs("data", exist_ok=True)
    generate(n=args.n).to_csv("data/diabetes.csv", index=False)
    train.run()
    fairness.run()

if __name__ == "__main__":
    main()
