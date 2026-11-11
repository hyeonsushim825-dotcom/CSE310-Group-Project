
import pandas as pd
import numpy as np

files = [
    "BenignTraffic.pcap.csv",
    "BenignTraffic1.pcap.csv",
    "BenignTraffic2.pcap.csv",
    "BenignTraffic3.pcap.csv"
]

protocols = [
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP",
    "SSH", "IRC", "TCP", "UDP", "DHCP",
    "ARP", "ICMP", "IGMP", "IPv", "LLC"
]

for file in files:
    df = pd.read_csv(file, nrows=10000)

    print(f"\n{'='*50}")
    print(file)
    print(f"{'='*50}")

    # 1. Protocol distribution
    print("\n[Protocol Distribution]")
    for col in protocols:
        if col in df.columns:
            count = (df[col] > 0).sum()
            percentage = count / len(df) * 100
            print(f"{col:10s}: {percentage:.2f}%")

    # 2. Feature variability
    print("\n[Feature Variability]")
    stats = pd.DataFrame({
        "Unique": df.nunique(),
        "Std": df.std(),
        "Zero_%": (df == 0).mean() * 100
    })

    print(stats.to_string())

    # 3. Constant features
    constant = df.columns[df.nunique() <= 1].tolist()
    print("\nConstant Features:", constant)
