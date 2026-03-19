#!/bin/bash
# Run this on Hetzner VPS as root BEFORE deploying Maskki
# Usage: ssh root@YOUR_IP 'bash -s' < scripts/harden-server.sh

set -e
echo "========================================="
echo "  Maskki Server Hardening Script"
echo "========================================="

# --- 1. Create deploy user (not root) ---
echo "[1/8] Creating deploy user..."
if ! id "maskki" &>/dev/null; then
    adduser --disabled-password --gecos "Maskki Deploy" maskki
    usermod -aG docker maskki
    usermod -aG sudo maskki

    # Copy SSH keys from root to maskki user
    mkdir -p /home/maskki/.ssh
    cp /root/.ssh/authorized_keys /home/maskki/.ssh/
    chown -R maskki:maskki /home/maskki/.ssh
    chmod 700 /home/maskki/.ssh
    chmod 600 /home/maskki/.ssh/authorized_keys

    echo "  User 'maskki' created with docker + sudo access"
else
    echo "  User 'maskki' already exists"
fi

# --- 2. SSH Hardening ---
echo "[2/8] Hardening SSH..."
SSH_CONFIG="/etc/ssh/sshd_config"
cp $SSH_CONFIG ${SSH_CONFIG}.backup

# Change port
SSH_PORT=2847  # Random port
sed -i "s/^#*Port .*/Port $SSH_PORT/" $SSH_CONFIG

# Disable root login
sed -i "s/^#*PermitRootLogin .*/PermitRootLogin no/" $SSH_CONFIG

# Disable password auth
sed -i "s/^#*PasswordAuthentication .*/PasswordAuthentication no/" $SSH_CONFIG

# Disable empty passwords
sed -i "s/^#*PermitEmptyPasswords .*/PermitEmptyPasswords no/" $SSH_CONFIG

# Limit auth attempts
sed -i "s/^#*MaxAuthTries .*/MaxAuthTries 3/" $SSH_CONFIG

# Timeout idle sessions
echo "ClientAliveInterval 300" >> $SSH_CONFIG
echo "ClientAliveCountMax 2" >> $SSH_CONFIG

systemctl restart sshd
echo "  SSH hardened (Port: $SSH_PORT, Root login: disabled, Password: disabled)"
echo "  SAVE THIS: New SSH command: ssh -p $SSH_PORT maskki@$(hostname -I | awk '{print $1}')"

# --- 3. Firewall ---
echo "[3/8] Configuring firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow $SSH_PORT/tcp comment "SSH"
ufw allow 80/tcp comment "HTTP (certbot)"
ufw allow 443/tcp comment "HTTPS (WebSocket)"
ufw --force enable
echo "  Firewall: only ports $SSH_PORT, 80, 443 open"

# --- 4. Fail2Ban ---
echo "[4/8] Installing fail2ban..."
apt install -y fail2ban > /dev/null 2>&1

cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = $SSH_PORT
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 86400
EOF

systemctl enable fail2ban
systemctl restart fail2ban
echo "  fail2ban installed (SSH: 3 attempts -> 24h ban)"

# --- 5. Auto-Updates ---
echo "[5/8] Enabling automatic security updates..."
apt install -y unattended-upgrades > /dev/null 2>&1
dpkg-reconfigure -f noninteractive unattended-upgrades
echo "  Automatic security updates enabled"

# --- 6. File Permissions ---
echo "[6/8] Setting secure file permissions..."
mkdir -p /opt/maskki
chown -R maskki:maskki /opt/maskki
chmod 750 /opt/maskki
echo "  /opt/maskki owned by maskki user (750)"

# --- 7. Kernel Hardening ---
echo "[7/8] Applying kernel security settings..."
cat >> /etc/sysctl.conf << EOF

# Maskki Security Hardening
net.ipv4.tcp_syncookies = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
EOF
sysctl -p > /dev/null 2>&1
echo "  Kernel hardened (SYN cookies, no redirects, no source routing)"

# --- 8. Summary ---
echo ""
echo "========================================="
echo "  HARDENING COMPLETE"
echo "========================================="
echo ""
echo "  SSH Port:    $SSH_PORT"
echo "  SSH User:    maskki"
echo "  SSH Command: ssh -p $SSH_PORT maskki@$(hostname -I | awk '{print $1}')"
echo "  Root Login:  DISABLED"
echo "  Password:    DISABLED"
echo "  Firewall:    $SSH_PORT, 80, 443 only"
echo "  Fail2Ban:    3 attempts -> 24h ban"
echo ""
echo "  IMPORTANT: Test SSH with new settings BEFORE closing this session!"
echo "  Open a NEW terminal and run:"
echo "  ssh -p $SSH_PORT maskki@$(hostname -I | awk '{print $1}')"
echo ""
