"""
SSL-Zertifikat Generator für Zwitscherkasten
Erstellt ein selbst-signiertes Zertifikat für HTTPS
"""

from OpenSSL import crypto
import os

def generate_certificate(ip_address="192.168.2.122"):
    """Generiert ein selbst-signiertes SSL-Zertifikat"""
    
    # Schlüsselpaar erstellen
    key = crypto.PKey()
    key.generate_key(crypto.TYPE_RSA, 2048)
    
    # Zertifikat erstellen
    cert = crypto.X509()
    cert.get_subject().C = "DE"
    cert.get_subject().ST = "Bayern"
    cert.get_subject().L = "Ingolstadt"
    cert.get_subject().O = "Zwitscherkasten"
    cert.get_subject().CN = ip_address
    
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)  # 1 Jahr gültig
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(key)
    cert.sign(key, 'sha256')
    
    # Dateien speichern
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
    
    print(f"✓ Zertifikat erstellt für IP: {ip_address}")
    print(f"  - cert.pem")
    print(f"  - key.pem")

if __name__ == "__main__":
    generate_certificate()
