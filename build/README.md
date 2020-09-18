Here you find a build for freeglut

These steps were followed to build freeglut on the following OS:
```
Distributor ID: Ubuntu
Description:  Ubuntu 16.04.7 LTS
Release:  16.04
```

1. Download freeglut latest version from [here](http://freeglut.sourceforge.net/)
2. Unzip the folder and cd to it
3. Run `cmake .`
4. Run `make`
5. Run `make install`

Note: You might need to set `CMAKE_INSTALL_PREFIX` to the target directory you want.
