import wellinfs.wellModel as wm
wellname = 'C02'
well = wm.well(wellname)
gridsLen = 20
well.setgrids(gridsLen)
well.showGrids