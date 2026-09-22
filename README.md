# New England Landsat mosaic, beside CTrees biomass

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/kentstephen/ne-landsat-ctrees-marimo/blob/main/ne-landsat-ctrees-pair.py)

One marimo notebook, two maps under one camera. On the left, the annual
leaf-on Landsat mosaic of New England, 2000 to 2025, one year at a time,
in true colour or NDVI. On the right, CTrees aboveground biomass folded
to H3 hexagons over the same view: the change between two years, or the
stock, with the uncertainty drawn as opacity. Click a hexagon and both
panels tell its story on the same 26 years: the mosaic's NDVI series and
where each year's pixels came from on the left, the biomass, its loss
year and its uncertainty on the right.

Nothing is read from disk. The mosaic and its boundary layers come from
the Source Coop bucket over http, CTrees from its public bucket on AWS.

## Run it

    uv run marimo edit ne-landsat-ctrees-pair.py

The notebook declares its own dependencies in its header, so `uv` builds
the environment on first run. Python 3.12 or later. The first view folds
CTrees for the box on screen, which takes a few seconds; after that a
change of window is a frame, not a fetch. The hexagons fold from zoom 9.

`docs/reading-the-pair.md` explains every control and what the fills and
the panels mean.

## The second notebook: the mosaic beside its own change

`ne-landsat-change-h3.py` is a fork of the pair with the right-hand map
swapped. It does not use CTrees data at all. Both panes come from the one
Landsat mosaic: the composite on the left, and on the right an H3 fill
folded from the same pyramid, showing where a pixel's index (NBR, NDVI or
NDMI) fell and stayed down between two years. The fold runs at 30 m when
zoomed in and coarsens as the view widens, down to res 11 hexagons.

    uv run marimo edit ne-landsat-change-h3.py

Same dependency header, same `uv` setup, Python 3.12 or later. Nothing is
read from disk. Set `NE_DATA` to point it at a local range-capable http
server over the same bucket layout (for example `python -m RangeHTTPServer`)
instead of Source Coop.

## The data

- The mosaic: [landsat-mosaics-new-england on Source Coop](https://source.coop/kentstephen/landsat-mosaics-new-england),
  a Zarr v3 multiscales pyramid of annual leaf-on composites of Landsat
  Collection 2 surface reflectance, 30 m to 3840 m, with a `source` plane
  that says which look each pixel came from. CC0-1.0. Built by
  [ne-landsat-temporal-mosaic](https://github.com/kentstephen/ne-landsat-temporal-mosaic).
- CTrees global aboveground biomass, 100 m, annual 2000 to 2025, with a
  residual standard error per pixel and year. CC-BY 4.0,
  doi 10.82924/7vmb-zv66; Yang, Saatchi et al. 2026. Read from its
  Icechunk store on AWS Open Data.
- Wildlands of New England GIS Data 1900-2022, Harvard Forest Data
  Archive HF435, Foster, Johnson and Hall 2023. CC0.
- The New England clip from TIGER/Line 2024, U.S. Census Bureau. Water
  bodies from the National Hydrography Dataset High Resolution, USGS. Both
  public domain. These ship beside the mosaic in the bucket's
  `supplemental/` folder with their own README.
- Overture Maps divisions (CDLA Permissive 2.0), read live from two Source
  Cooperative repositories: state and county boundaries as PMTiles from
  cboettig/overturemaps (release 2026-02-18.0), and the town under a click
  from the GeoParquet in fused/overture (release 2026-05-20-0).
- Basemap tiles by OpenFreeMap, place search by Photon (komoot), both on
  OpenStreetMap data (ODbL).

## License

MIT.
