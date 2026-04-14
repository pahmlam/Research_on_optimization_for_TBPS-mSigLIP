#!/bin/bash
# Collect system information for Qualcomm RB3 Gen2

OUTPUT_FILE="system_info.md"

echo "# System Information" > $OUTPUT_FILE
echo "## Qualcomm RB3 Gen2 Development Kit" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "**Generated:** $(date)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## Hardware Information" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "### CPU" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
lscpu >> $OUTPUT_FILE 2>/dev/null || cat /proc/cpuinfo >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "### Memory" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
free -h >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "### Storage" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
df -h >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## Software Information" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "### OS" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
cat /etc/os-release >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "### Python" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
python3 --version >> $OUTPUT_FILE 2>&1
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "### SNPE/QNN Packages" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
dpkg -l | grep -E "snpe|qnn" >> $OUTPUT_FILE 2>/dev/null
echo '```' >> $OUTPUT_FILE

echo "System info saved to $OUTPUT_FILE"
