
# Redistricting 


This project aims to tackle redistricting problem and improve upon existing algorihms. Our approach combines advanced applied algebra with machine learning algorithms to precisely solve the problem on large-scale datasets.


Joint work with Ivan Gvozdanovic



## Data 


1. Test dataset gerrymandria.json by [Gerrychain](https://github.com/mggg/GerryChain/blob/main/docs/_static/gerrymandria.json)


2. 2020 Census Redistricting Data (P.L. 94-171) Census Tract Shapefile for Illinois by [Redistricting Data Hub]{https://redistrictingdatahub.org/dataset/illinois-census-tract-boundaries-2020}.

##Retrieval Date 02/25/2021
Source: https://www.census.gov/programs-surveys/decennial-census/about/rdo/summary-files.html#P1

##Fields
    Field Name                                                                                                                                                                   Description
0    STATEFP20                                                                                                                                                   2020 Census state FIPS code
1   COUNTYFP20                                                                                                                                                  2020 Census county FIPS code
2    TRACTCE20                                                                                                                                                        2020 Census tract code
3      GEOID20                                                              Census tract identifier; a concatenation of 2020 Census state FIPS code, county FIPS code, and census tract code
4       NAME20  2020 Census tract name, this is the census tract code converted to an integer or integer plus 2-character decimal if the last two characters of the code are not both zeros.
5   NAMELSAD20                                                                                           2020 Census translated legal/statistical area description and the census tract name
6      MTFCC20                                                                                                                                          MAF/TIGER feature class code (G5020)
7   FUNCSTAT20                                                                                                                                                 2020 Census functional status
8      ALAND20                                                                                                                                                         2020 Census land area
9     AWATER20                                                                                                                                                        2020 Census water area
10  INTPTLAT20                                                                                                                                    2020 Census latitude of the internal point
11  INTPTLON20                                                                                                                                   2020 Census longitude of the internal point



## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
