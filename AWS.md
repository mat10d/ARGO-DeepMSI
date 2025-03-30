# Data Transfer Guide - EC2 with Windows and EBS

This guide explains how to set up and transfer data to an EC2 instance with Windows Server and an attached EBS volume.

## Prerequisites

- An EC2 instance running Windows Server
- An EBS volume attached to the instance
- Administrative access to the EC2 instance
- Source data files ready for transfer
- SSH client on your local machine

## 1. Enable SSH on Windows Server

Follow these steps to install and configure the SSH server on your Windows instance. These instructions are adapted from [Emmanuel Tsouris's guide](https://www.emmanueltsouris.com/posts/enable-ssh-server-windows-server-2022/).

### Step 1: Install OpenSSH Server

1. Open Windows Settings:
   * Press `Win + I` to open the Settings app
2. Navigate to Apps:
   * Go to `Apps` > `Optional Features`
3. Add a Feature:
   * Click on `Add a feature`
4. Install OpenSSH Server:
   * In the search box, type "OpenSSH Server" and select it
   * Click `Install`

### Step 2: Start and Configure the SSH Service

1. Open PowerShell as Administrator:
   * Right-click on the Start button and select `Windows PowerShell (Admin)`
2. Start the SSH Service:
   ```powershell
   Start-Service sshd
   ```
3. Set SSH to Start Automatically:
   ```powershell
   Set-Service -Name sshd -StartupType 'Automatic'
   ```
4. Configure the Firewall Rule:
   ```powershell
   New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
   ```

### Step 3: Verify SSH Configuration

1. Check the SSH Service Status:
   ```powershell
   Get-Service -Name sshd
   ```
2. The service should show as "Running"

## 2. Configure EC2 Security Group

Add an inbound rule to allow SSH access:

1. Open the EC2 Console
2. Select your instance's Security Group
3. Add an inbound rule:
   - Type: SSH
   - Protocol: TCP
   - Port Range: 22
   - Source: Your IP address or appropriate CIDR range
   - Description: SSH access

## 3. Connect and Transfer Files

Use SFTP to transfer files to your EC2 instance:

```bash
# Connect using SFTP with the username account and IP address
# Password will be decrypted password from uploading PEM key to AWS
sftp username@your-instance-ip 

# Navigate to the EBS volume and create msi-data directory
sftp> cd /E:
sftp> mkdir msi-data

# Exit SFTP and use scp -r to transfer the entire data directory
exit
scp -r /lab/barcheese01/mdiberna/ARGO-DeepMSI/data/* username@your-instance-ip:/E:/msi-data/

# To verify the transfer:
sftp ADMINISTRATOR@your-instance-ip
sftp> cd /E:/data
sftp> ls
LASUTH  LUTH  OAUTHC  retrospective_msk  retrospective_oau  UITH

# To download the entire directory structure back:
exit
scp -r username@your-instance-ip:/E:/data/* /path/to/local/destination/

# List contents recursively to verify directory structure
ls -R /path/to/local/destination/

# Navigate to your data directory
cd /path/to/downloaded/msi-data

# Create raw subfolder in each dataset directory and move files
for dataset in */; do
  # Create raw directory if it doesn't exist
  mkdir -p "${dataset}raw"
  
  # Find all files (not directories) in the dataset folder and move them to raw/
  find "${dataset}" -maxdepth 1 -type f -exec mv {} "${dataset}raw/" \;
done

# Verify the structure
ls -la */raw/
```

### Important Notes

- Replace `your-instance-ip` with your EC2 instance's public IP address
- Replace `username` as the username (note the uppercase)
- The password is your Windows Administrator password (can be decrypted from AWS Console)
- File transfers are encrypted over SSH
- Large files may take significant time to transfer
- The EBS volume is typically mounted as `E:` drive, but verify this on your instance

## 4. Download Slide Tables from Halo

To export slide tables from Halo:

1. Log in to your Halo account

2. Navigate to the study:
   * Select the appropriate study from your dashboard

3. Export slide data:
   * Click on the study name
   * Click "File export" in the menu options
   * Select CSV format for the export
   * Choose the data fields you want to include
   * Click "Export" to generate the file
   * Download the exported CSV file to your local machine

4. Organize the exported CSV files:
   * Name the files consistently (e.g., `halo_link_[CENTER]_export.csv`)
   * Store these files in a location where they can be easily referenced alongside your image data

5. Verify the slide tables contain the necessary information:
   * Check that slide IDs match your image files
   * Ensure all required metadata fields are present
   * Validate that the data is complete for all centers

These slide tables will provide essential metadata linking your image files to clinical information needed for your analysis.

## 5. Best Practices

- Use a stable internet connection for large file transfers
- Consider using AWS CLI for bulk transfers
- Monitor EBS volume usage to ensure sufficient space
- Keep your instance's security group rules as restrictive as possible
- Regularly backup important data

## 6. Troubleshooting

### Common Issues and Solutions

1. Connection refused
   - Verify SSH service is running using `Get-Service -Name sshd`
   - Check security group rules in AWS Console
   - Confirm instance is running

2. Authentication failed
   - Verify administrator password from AWS Console
   - Ensure using correct username format (ADMINISTRATOR@ip-address)
   - Check that you're using the latest password if recently changed

3. Cannot access EBS volume
   - Verify volume is properly mounted in Windows Disk Management
   - Check volume permissions in Windows
   - Ensure volume is initialized and formatted

## 7. Additional Resources

- [Original SSH Setup Guide by Emmanuel Tsouris](https://www.emmanueltsouris.com/posts/enable-ssh-server-windows-server-2022/)
- [AWS Documentation - Connect to Windows Instance](https://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/connecting_to_windows_instance.html)
- [AWS Documentation - EBS Volumes](https://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/ebs-volumes.html)
- [OpenSSH Documentation](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse)

## 8. Alternative Transfer Methods

If SFTP isn't suitable, consider these alternatives:

- AWS CLI (aws s3 cp)
- RDP with local drive mapping
- AWS DataSync
- AWS Transfer Family