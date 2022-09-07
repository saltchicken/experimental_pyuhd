# experimental_pyuhd
## Install

### Build UHD from source
```
sudo apt-get install autoconf automake build-essential ccache cmake cpufrequtils doxygen ethtool g++ git inetutils-tools \
libboost-all-dev libncurses5 libncurses5-dev libusb-1.0-0 libusb-1.0-0-dev libusb-dev \
python3-dev python3-mako python3-numpy python3-requests python3-scipy python3-setuptools python3-docutils python3-ruamel.yaml

git clone https://github.com/EttusResearch/uhd.git
cd uhd/host
mkdir build
cd build
cmake -DENABLE_TESTS=OFF -DENABLE_C_API=OFF -DENABLE_MANUAL=OFF ..
make -j8
sudo make install
sudo ldconfig

sudo uhd_images_downloader

cd <install-path>/lib/uhd/utils
sudo cp uhd-usrp.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```
#### Verify in Python
```Python
import uhd
```
If no errors, you are good to go

If 'module UHD' not found.
```
sudo find / -name uhd
```
find the line that contains 'dist-packages' (In my case /usr/local/local/lib/python3.10/dist-packages/uhd)
```
export PYTHONPATH=/usr/local/local/lib/python3.10/dist-packages 
```
(Notice that 'uhd' was not included)
Add this line to your ~/.bashrc file to export on startup
Verify if this fixed the issue.

### Build SoapySDR from source
```
sudo apt-get install cmake g++ libpython3-dev python3-numpy swig
git clone https://github.com/pothosware/SoapySDR.git
cd SoapySDR
git pull origin master
mkdir build
cd build
cmake ..
make -j4
sudo make install
sudo ldconfig #needed on debian systems
SoapySDRUtil --info
```
##### You will also need to build the Soapy modules for your specific device
Reference for HackRF: https://github.com/pothosware/SoapyHackRF/wiki#building-soapy-hack-rf

### PIP install required python modules
```
sudo apt install python3-pip
pip install matplotlib scipy
```
