

# Telemetry API
## Sensor Variables
-10|ENV__SOIL__VWC - volumetric soil water content at 10 cm depth (%)
-30|ENV__SOIL__VWC - volumetric soil water content at 30 cm depth (%)
-45|ENV__SOIL__VWC - volumetric soil water content at 45 cm depth (%)
-10|ENV__SOIL__T - soil temperature at 10 cm depth (°C)
-30|ENV__SOIL__T - soil temperature at 30 cm depth (°C)
-45|ENV__SOIL__T - soil temperature at 45 cm depth (°C)
TOP|ENV__ATMO__T - temp erdoberfläche
DOC|ENV__SOIL__IRRIGATION - Irrigation Menge

### All
timestamp     ,
datetime            ,
DEVICE|DEV__ENERGY__VCAP ,
DEVICE|DEV__ALERT__TRIGGERED ,
DEVICE|DEV__RF__RSSI ,
DEVICE|DEV__RF__SINR ,
+15|ENV__ATMO__T   ,
-10|ENV__SOIL__T ,
-30|ENV__SOIL__T ,
-45|ENV__SOIL__T ,
-10|ENV__SOIL__CAPACITANCE__ABSOLUTE ,
-30|ENV__SOIL__CAPACITANCE__ABSOLUTE ,
-45|ENV__SOIL__CAPACITANCE__ABSOLUTE ,
DEVICE|DEV__SOILSENSOR__ID ,
-10|ENV__SOIL__NORM_CAP ,
-10|ENV__SOIL__NORM_ER ,
-10|ENV__SOIL__VWC ,
-30|ENV__SOIL__NORM_CAP ,
-30|ENV__SOIL__NORM_ER ,
-30|ENV__SOIL__VWC ,
-45|ENV__SOIL__NORM_CAP ,
-45|ENV__SOIL__NORM_ER ,
-45|ENV__SOIL__VWC ,
TOP|ENV__ATMO__T   ,
OVERALL|ENV__SOIL__MOISTURE_LEVEL__COARSE ,
-10|ENV__SOIL__MOISTURE_LEVEL__COARSE ,
-30|ENV__SOIL__MOISTURE_LEVEL__COARSE ,
-45|ENV__SOIL__MOISTURE_LEVEL__COARSE ,
-10|ENV__SOIL__MOISTURE_LEVEL__FINE ,
-30|ENV__SOIL__MOISTURE_LEVEL__FINE ,
-45|ENV__SOIL__MOISTURE_LEVEL__FINE ,
TOP|ENV__ATMO__IRRADIATION ,
TOP|ENV__ATMO__RADIANT_EXPOSURE ,
OVERALL|ENV__SOIL__MOISTURE_LEVEL__FINE ,
TOP|ENV__ATMO__FROST_LEVEL__COARSE ,
TOP|ENV__ATMO__FROST_LEVEL__FINE ,
OVERALL|ENV__ATMO__FROST_LEVEL__COARSE ,
OVERALL|ENV__ATMO__FROST_LEVEL__FINE ,
MTB|ENV__ATMO__T ,
MTB|ENV__ATMO__RAIN__DELTA ,
MTB|ENV__ATMO__RH ,
MTB|ENV__ATMO__P ,
MTB|ENV__ATMO__WIND__DIRECTION ,
MTB|ENV__ATMO__WIND__SPEED ,
MTB|ENV__SOIL__ET ,
MTB|ENV__SOIL__T ,
MTB|ENV__SOIL__VWC ,
MTB|CROP__LEAF__LWI ,
MTB|ENV__SOIL__PET ,
MTB|ENV__ATMO__DEWPOINT ,
MTB|ENV__SOIL__ET0 ,
DOC|ENV__SOIL__IRRIGATION ,
TOP|ENV__ATMO__T__DRY ,
TOP|ENV__ATMO__T__WET




## Weather Sensor Variables

TOP|ENV__ATMO__T - (dry) temperature (°C) (also accessible via ENV__ATMO__T__DRY)
TOP|ENV__ATMO__RH - relative humidity (%) vllt. hilfreich
TOP|ENV__ATMO__IRRADIATION - solar radiation (W/m²) relevant
EXT1|ENV__ATMO__RAIN

### All
timestamp,
datetime,
DEVICE|DEV__ENERGY__VCAP,
DEVICE|DEV__ALERT__TRIGGERED,
DEVICE|DEV__RF__RSSI,
DEVICE|DEV__RF__SINR,
TOP|ENV__ATMO__T,
TOP|ENV__ATMO__RH,
EXT1|ENV__ATMO__RAIN,
TOP|ENV__ATMO__IRRADIATION,
TOP|ENV__ATMO__RADIANT_EXPOSURE,
TOP|ENV__ATMO__P,
TOP|ENV__ATMO__FROST_LEVEL__COARSE,
TOP|ENV__ATMO__FROST_LEVEL__FINE,
OVERALL|ENV__ATMO__FROST_LEVEL__COARSE,
OVERALL|ENV__ATMO__FROST_LEVEL__FINE,
MTB|ENV__ATMO__T,
MTB|ENV__ATMO__RAIN__DELTA,
MTB|ENV__ATMO__RH,
MTB|ENV__ATMO__P,
MTB|ENV__ATMO__WIND__DIRECTION,
MTB|ENV__ATMO__WIND__SPEED,
MTB|ENV__SOIL__ET,
MTB|ENV__SOIL__T,
MTB|ENV__SOIL__VWC,
MTB|CROP__LEAF__LWI,
MTB|ENV__SOIL__PET,
MTB|ENV__ATMO__DEWPOINT,
MTB|ENV__SOIL__ET0,
EXT|ENV__ATMO__RAIN__DELTA,
TOP|ENV__ATMO__T__DRY,
TOP|ENV__ATMO__T__WET,
TEST|ENV__ATMO__T,
TEST2|ENV__ATMO__T,
TEST|ENV__ATMO__RH,
TEST2|ENV__ATMO__RH,
RAW|TOP|ENV__ATMO__T


# Find Latest API

## Attributes
REC|ENV__SOIL__IRRIGATION
-10|ENV__SOIL__VWC__fc
-45|ENV__SOIL__VWC__fc
ENV__ATMO__RAIN__24h
-30|ENV__SOIL__VWC__fc


## Forecast
TOP|ENV__ATMO__T - (dry) temperature (°C) (also accessible via ENV__ATMO__T__DRY)
EXT|ENV__ATMO__RAIN__DELTA - rain amounts between current and next measurement (mm)
TOP|ENV__ATMO__RH - relative humidity (%)
TOP|ENV__ATMO__P - air pressure (hPa)
EXT|ENV__ATMO__WIND__DIRECTION - wind direction (°)
EXT|ENV__ATMO__WIND__SPEED - wind speed (km/h)
TOP|ENV__SOIL__ET 
-10|ENV__SOIL__T - soil temperature at 10 cm depth (°C)
TOP|CROP__LEAF__LWI 
TOP|ENV__SOIL__PET
TOP|ENV__ATMO__DEWPOINT
TOP|ENV__SOIL__ET0

## All Variables

latitude - latitude of the device location

longitude - longitude of the device location

name - serial number of the device

label - custom device name

REC|ENV__SOIL__IRRIGATION - current irrigation recommendation at the device location (mm)

ENV__ATMO__RAIN__24h - rain sum for the last 24 hours (mm)

ENV__ATMO__RAIN__3d - rain sum for the last 3 days (mm)

ENV__ATMO__RAIN__7d - rain sum for the last 7 days (mm)

ENV__ATMO__RAIN__14d - rain sum for the last 14 days (mm)

ENV__ATMO__RAIN__30d - rain sum for the last 30 days (mm)

EXT|ENV__ATMO__RAIN__DELTA - rain amounts between current and next measurement (mm)

TOP|ENV__ATMO__T - (dry) temperature (°C) (also accessible via ENV__ATMO__T__DRY)

TOP|ENV__ATMO__T__WET - wet temperature (°C)

TOP|ENV__ATMO__RH - relative humidity (%)

TOP|ENV__ATMO__P - air pressure (hPa)

TOP|ENV__ATMO__IRRADIATION - solar radiation (W/m²)

EXT|ENV__ATMO__WIND__SPEED - wind speed (km/h)

EXT|ENV__ATMO__WIND__SPEED__PEAK - wind speed peak (km/h)

EXT|ENV__ATMO__WIND__DIRECTION - wind direction (°)

-10|ENV__SOIL__VWC - volumetric soil water content at 10 cm depth (%)

-30|ENV__SOIL__VWC - volumetric soil water content at 30 cm depth (%)

-45|ENV__SOIL__VWC - volumetric soil water content at 45 cm depth (%)

-10|ENV__SOIL__VWC__fc - forecast (upcoming 7 days) of volumetric soil water content at 10 cm depth (%)

-30|ENV__SOIL__VWC__fc - forecast (upcoming 7 days) of volumetric soil water content at 30 cm depth (%)

-45|ENV__SOIL__VWC__fc - forecast (upcoming 7 days) of volumetric soil water content at 45 cm depth (%)

-10|ENV__SOIL__T - soil temperature at 10 cm depth (°C)

-30|ENV__SOIL__T - soil temperature at 30 cm depth (°C)

-45|ENV__SOIL__T - soil temperature at 45 cm depth (°C)