# pi-projector
Development of the projector preprocessing (pixels processing) for the raspberry PI. (Academic project)

## Setup

- Install wiring pi by putting the `WiringPi` folder into src. When opening it, it should lead to the root of the git clone of the repository of the library.
- Install opencv and ffmpeg using your distribution's package manager. (such as apt)


## Compiling

### With WiringPi

- `./compile_parall_wiring_pi.sh`

### Without WiringPi

- `./compile_parall.sh`

## Usage

### Full process

- `run.sh <video_path>`

> **Note:** Please avoid spaces or Unicode characters.

### Program only

- `./read_pixels_parall.out ./tmp/projector_input.mp4 [flags]`

The list of available flags can be obtained with:
- `./read_pixels_parall.out`
Or in the source code: `./src/read_pixels_parall.cpp`

> **Note:** Other files are for test purposes only.

## Bus

SPI - 2MHz

## Licence

```
    pi-projector is a program to parse a video file and send its processed pixels to an ESP32 microcontroller
    Copyright (C) 2023-2024  Ikram Achali, Charly Schmidt alias Picorims<picorims.contact@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
```