import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.80.154', 22, 'ws', 'ws', timeout=4)

cmd = 'ps -fp 1414; ls -la /home/ws/ugv_rpi/.greet 2>/dev/null || echo "greet unlinked"'
stdin, stdout, stderr = ssh.exec_command(cmd)
print("STATUS:\n", stdout.read().decode('utf-8', errors='ignore'))
ssh.close()
