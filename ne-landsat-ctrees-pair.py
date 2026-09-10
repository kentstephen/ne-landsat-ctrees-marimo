# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql>=0.3.3",
#     "xarray",
#     "zarr>=3",
#     "icechunk",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "anywidget>=0.9",
#     "traitlets",
#     "numpy",
#     "duckdb>=1.5.5",
#     "pillow",
#     "geopandas>=1.1",
#     "shapely>=2",
#     "rasterio",
#     "obstore>=0.11",
# ]
# ///


"""The New England leaf-on Landsat mosaic beside one H3 fill of CTrees biomass.

A single-file marimo notebook. Nothing is read from disk: the multiscales
pyramid and the supplemental layers come from the Source Coop bucket over
http (DATA, below), CTrees from its own public bucket on AWS. The grid and
the tile renderer are the cells after the imports.

The picture is the mosaic, and the labels are CTrees biomass folded to H3
inside the New England AOI and nowhere else.

Two maps in one widget, one camera.

LEFT, the picture, never covered: the leaf-on mosaic, one year at a time
(2000..2025), true colour or NDVI, on one slider. Tiles are rendered in the
kernel from the pyramid: at each zoom the level whose pixel is the finest
not smaller than the screen pixel, so a zoomed-out view is a few chunks.

RIGHT, one opaque H3 fill per hexagon at res 9 and coarser, from

  CTrees global aboveground biomass (AWS Open Data, Icechunk): 100 m,
  annual 2000-2025, Mg/ha, with a residual standard error per pixel per
  year. All 26 years are read for the box at once; the window (from year,
  to year) picks two ends, and the fills are the change between the ends
  (blue gain, orange loss) and the stock at the to-end. The year the
  biomass first fell past a threshold is in the click readout, not a fill.

The read box is clamped to the AOI (grid.WEST..EAST, SOUTH..NORTH), so the
fold never reaches for CTrees pixels outside New England. Both panes are
then clipped to the six states: supplemental/tiger_states.parquet (raw
Census TIGER 2024, to the 3 nautical mile limit, dissolved). The store
itself is not masked, its tiles are whole 4096 px squares that keep NY,
Quebec, New Brunswick and the ocean; the clip lives here. The picture goes
transparent outside the polygon (tiles.set_clip); the fold averages only
the CTrees samples inside it and drops a hexagon when fewer than half its
samples are inside, so a cell straddling the coast or the NY line is drawn
when it is mostly New England.

Water is masked the same way, at the pixel level, from
supplemental/nhd_water_bodies.parquet (NHD HR lakes, ponds, reservoirs, estuaries, bays
and wide rivers, 1 ha and up). CTrees carries a value over open water, zero
or near it with an uncertainty, not a nodata flag, so without the mask the
fold draws hexagons over lakes and a shore hexagon averages lake zeros in
with its forest. With it, a hexagon over open water is not drawn and a
shore hexagon is the mean of its land pixels. WATER MASK (the button, or
`m`) is on by default and toggles in the kernel: the fold is re-run for the
held box. ADMIN (`s`) draws the state and county boundaries on both panes
from Overture Maps divisions, read live from Source Cooperative as PMTiles
(cboettig/overturemaps): the browser range-reads the tiles it needs and the
kernel never sees them. The basemap's own admin lines are hidden, so these
are the only boundaries on the map. A click names its town, county and
state: the state and county come from those same tiles in the browser, the
town from one DuckDB point query against the Overture divisions GeoParquet
on Source Cooperative (fused/overture), off the loop, 1 to 2 s.

The uncertainty is opacity: a change smaller than the uncertainty of its
two ends is drawn faint.

WILDLANDS (the button, or `w`): the 426 Wildlands of New England, 2022,
boundaries only, one opaque orange line on both panes, never filled. Off by
default. The whole
layer is 426 polygons, so it crosses to the browser once when the map is
built and the toggle never touches the kernel.

A click fills both panels: the mosaic's own NDVI series under the left pane,
CTrees' biomass story under the right, on the same 26 years.

The fold is an H3 UDF inside DataFusion: the pixels cross as one Dataset
and the cell is the GROUP BY. Nothing is tessellated in the kernel.

Run: uv run marimo edit ne-landsat-ctrees-pair.py

Attribution: CTrees global AGB (CC-BY 4.0, doi 10.82924/7vmb-zv66; Yang,
Saatchi et al. 2026). Wildlands of New England GIS Data 1900-2022
(CC0), Harvard Forest Data Archive HF435, Foster, Johnson and Hall 2023.
The mosaic is USGS Landsat
Collection 2 Level-2, built by github.com/kentstephen/ne-landsat-temporal-mosaic. Place search by Photon
(komoot), OpenStreetMap data (ODbL).
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", sql_output="native")


@app.cell
def _():
    import asyncio
    import json
    import math
    import time
    import traceback

    import numpy as np
    import pyarrow as pa
    import xarray as xr
    import icechunk
    import duckdb
    import marimo as mo
    import anywidget
    import traitlets

    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells

    return (
        XarrayContext,
        anywidget,
        asyncio,
        coordinates_to_cells,
        duckdb,
        icechunk,
        json,
        math,
        mo,
        np,
        pa,
        time,
        traceback,
        traitlets,
        udf,
        xr,
    )


@app.cell
def _():
    # ---- where the data lives ------------------------------------------------------
    # One bucket holds the pyramid and the supplemental layers, read over http
    # by URL; nothing comes from disk. NE_DATA overrides the bucket: a local
    # range-capable server over the repo root serves the same layout
    # (`python -m RangeHTTPServer`; the stdlib http.server has no ranges).
    import os as _os
    import io as _io
    import obstore as _obstore
    from obstore.store import HTTPStore as _HTTPStore

    DATA = _os.environ.get("NE_DATA", "https://data.source.coop/kentstephen/landsat-mosaics-new-england").rstrip("/")
    PYRAMID = f"{DATA}/pyramid_v1"
    SUPP = f"{DATA}/supplemental"

    # Both reads go through obstore's http client (Rust, its own connection
    # pool): the pyramid as a zarr ObjectStore over it, the parquet layers as
    # one GET each. Note data.source.coop sits behind Cloudflare, which
    # answers 403 to the stdlib urllib user agent; obstore's goes through.
    def http_store(url):
        return _HTTPStore(url)

    def read_gpq(url, columns=None):
        """A GeoParquet on the bucket as a GeoDataFrame. The layers are 1 to
        19 MB, so the whole file comes down in one GET and pyarrow reads it
        from memory; no range requests, no filesystem plumbing."""
        import geopandas as gpd

        base, _, name = url.rpartition("/")
        data = bytes(_obstore.get(http_store(base), name).bytes())
        return gpd.read_parquet(_io.BytesIO(data), columns=columns)

    return PYRAMID, SUPP, http_store, read_gpq


@app.cell
def _():
    # ---- grid: the pyramid's target grid (src/landsat_mosaic/grid.py, the part used) --
    # EPSG:4326, 0.00025 degree pixels, north-up, origin at the north-west
    # corner of the New England bbox. Level 0 of the pyramid is this grid;
    # level k is it at 2**k pixels.
    from types import SimpleNamespace as _NS

    grid = _NS(RES=0.00025, WEST=-73.75, NORTH=47.5, HEIGHT=26200, WIDTH=27600)
    grid.SOUTH, grid.EAST = grid.NORTH - grid.HEIGHT * grid.RES, grid.WEST + grid.WIDTH * grid.RES
    return (grid,)


@app.cell
def _(PYRAMID, grid, http_store, math, np, time):
    # ---- tiles: Web Mercator PNGs from the pyramid (src/landsat_mosaic/tiles.py) ----
    # For a tile at zoom z the level whose pixel is the finest one not smaller
    # than the tile's pixel is read (1 to 2 source pixels per screen pixel), so
    # a zoomed-out map costs a few 512 chunks instead of a whole year's level
    # 0. The store is opened once, over http, and nothing else is cached.
    # Modes: `tc` true colour under a gamma, `ndvi` on the fixed NDVI_LO..NDVI_HI
    # scale (Carto Emrld, dark low, bright high). Nodata (all bands 0) is
    # transparent, and so is everything outside the clip (`set_clip`).
    import io as _io
    import threading as _threading
    from types import SimpleNamespace as _NS

    import zarr as _zarr
    from PIL import Image as _Image

    tiles = _NS()
    tiles.GROUP = "leafon"
    tiles.YEARS = list(range(2000, 2026))
    tiles.NLEVELS = 8
    tiles.SCALE, tiles.OFFSET = 0.0000275, -0.2
    tiles.REFL_MAX = 0.3      # reflectance drawn as white at gain 1 (as the quicklooks)
    # True colour is a gamma curve, not a linear stretch: leaf-on forest sits at
    # 0.02 to 0.04 reflectance, a tenth of REFL_MAX, so linear it draws at 12%
    # luminance and only a gain that clips fields and sand can lift it. At 2.0
    # forest comes up to about a third with nothing clipped; the gain slider
    # still multiplies underneath it. NDVI is a LUT and untouched.
    tiles.GAMMA = 2.0
    tiles.T = 256             # tile pixels
    tiles.MIN_Z, tiles.MAX_Z = 4, 13
    tiles.MODES = {"tc": ("red", "green", "blue"), "ndvi": ("nir", "red")}
    tiles.NDVI_LO, tiles.NDVI_HI = -0.1, 0.9
    _EMRLD = ("#074050", "#105965", "#217a79", "#4c9b82", "#6cc08b", "#97e196", "#d3f2a3")
    _R = 6378137.0
    _T = tiles.T

    def _lut(stops, n=256):
        pts = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in stops], np.float32)
        xs = np.linspace(0, 1, len(stops))
        t = np.linspace(0, 1, n)
        return np.stack([np.interp(t, xs, pts[:, k]) for k in range(3)], 1).astype(np.uint8)

    tiles.NDVI_LUT = _lut(_EMRLD)
    tiles.NDVI_HEX = ["#%02x%02x%02x" % tuple(int(v) for v in tiles.NDVI_LUT[i]) for i in range(0, 256, 17)]

    _lock = _threading.Lock()
    _geos = _threading.Lock()
    from collections import OrderedDict as _OD
    from concurrent.futures import ThreadPoolExecutor as _TPE, Future as _Future
    _bands = _TPE(max_workers=24, thread_name_prefix="band")

    # ---- the block cache: decoded 512 x 512 chunks, by (level, band, year, i, j)
    # The pyramid's chunks are 512 square at every level, so a chunk is a
    # block and a tile is the blocks under it. Cached decoded, they are what
    # makes a coarse tile a quarter of the fetches (four tiles a level down
    # share its blocks), lets neighbouring tiles share edges, and gives the
    # click's 26-year series its pixels for free where the view already
    # covered them. 0.5 MB a block; BLOCK_CACHE blocks, oldest out first.
    # Concurrent asks for one block share a single fetch.
    _BLK = 512
    _blocks = _OD()
    _inflight = {}
    _blk_lock = _threading.Lock()
    tiles.BLOCK_CACHE = 400
    tiles.block_hits = tiles.block_fetches = 0

    def _block(lv, band, t, bi, bj):
        key = (lv, band, t, bi, bj)
        with _blk_lock:
            v = _blocks.get(key)
            if v is not None:
                _blocks.move_to_end(key)
                tiles.block_hits += 1
                return v
            fut = _inflight.get(key)
            owner = fut is None
            if owner:
                fut = _inflight[key] = _Future()
        if not owner:
            return fut.result()
        try:
            a = array(lv, band)[t, bi * _BLK:(bi + 1) * _BLK, bj * _BLK:(bj + 1) * _BLK]
            with _blk_lock:
                _blocks[key] = a
                _blocks.move_to_end(key)
                while len(_blocks) > tiles.BLOCK_CACHE:
                    _blocks.popitem(last=False)
                _inflight.pop(key, None)
                tiles.block_fetches += 1
            fut.set_result(a)
            return a
        except BaseException as e:
            with _blk_lock:
                _inflight.pop(key, None)
            fut.set_exception(e)
            raise

    def regions(lv, asks):
        """asks: [(band, t, r0, r1, c0, c1)] -> [uint16 (r1-r0, c1-c0)], every
        block behind them fetched side by side through the cache."""
        keys = []
        for band, t, r0, r1, c0, c1 in asks:
            for bi in range(r0 // _BLK, (r1 - 1) // _BLK + 1):
                for bj in range(c0 // _BLK, (c1 - 1) // _BLK + 1):
                    keys.append((lv, band, t, bi, bj))
        keys = list(dict.fromkeys(keys))
        got = dict(zip(keys, _bands.map(lambda k: _block(*k), keys)))
        out = []
        for band, t, r0, r1, c0, c1 in asks:
            a = np.zeros((r1 - r0, c1 - c0), np.uint16)
            for bi in range(r0 // _BLK, (r1 - 1) // _BLK + 1):
                for bj in range(c0 // _BLK, (c1 - 1) // _BLK + 1):
                    blk = got[(lv, band, t, bi, bj)]
                    R0, C0 = bi * _BLK, bj * _BLK
                    rr0, rr1 = max(r0, R0), min(r1, R0 + blk.shape[0])
                    cc0, cc1 = max(c0, C0), min(c1, C0 + blk.shape[1])
                    if rr1 > rr0 and cc1 > cc0:
                        a[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = blk[rr0 - R0:rr1 - R0, cc0 - C0:cc1 - C0]
            out.append(a)
        return out

    tiles.regions = regions
    _state = {"root": None, "opened": 0.0}
    _clip = {"prep": None, "cache": {}}
    _MISS = object()
    _CLIP_CACHE = 4096

    def set_clip(geom):
        """Clip rendered tiles to `geom` (shapely (Multi)Polygon, EPSG:4326), or
        None to draw whole tiles again."""
        import shapely

        with _lock:
            _clip["prep"] = None
            _clip["cache"] = {}
            _png.clear()
            if geom is not None:
                shapely.prepare(geom)
                _clip["prep"] = geom

    def _inside(z, x, y):
        """(T, T) bool, True where the pixel centre is inside the clip; True
        everywhere when no clip is set. Whole-tile answers come from the tile
        box (inside, or disjoint), so the point test runs only on edge tiles,
        and each tile's answer is kept."""
        import shapely

        prep = _clip["prep"]
        if prep is None:
            return np.ones((_T, _T), bool)
        key = (z, x, y)
        got = _clip["cache"].get(key)
        if got is not None:
            return got
        lon, lat, (W, S, E, N) = _centers(z, x, y)
        box = shapely.box(W, S, E, N)
        # one thread at a time on the prepared geometry: GEOS prepared
        # predicates are not safe to run concurrently, and with tiles rendered
        # from a dozen threads the kernel segfaulted in contains (2026-09-09,
        # faulthandler). The test is microseconds; the lock costs nothing.
        with _geos:
            if prep.contains(box):
                m = np.ones((_T, _T), bool)
            elif prep.disjoint(box):
                m = np.zeros((_T, _T), bool)
            else:
                LON, LAT = np.meshgrid(lon, lat)
                m = shapely.contains_xy(prep, LON.ravel(), LAT.ravel()).reshape(_T, _T)
        with _lock:
            if len(_clip["cache"]) >= _CLIP_CACHE:
                _clip["cache"].clear()
            _clip["cache"][key] = m
        return m

    def _open():
        """The pyramid root, read only, opened once: zarr's ObjectStore over
        obstore's http client; the consolidated metadata is one fetch."""
        from zarr.storage import ObjectStore as _ObjectStore
        with _lock:
            if _state["root"] is None:
                _state["root"] = _zarr.open_group(_ObjectStore(http_store(PYRAMID), read_only=True), mode="r")
                _state["opened"] = time.time()
            return _state["root"]

    def array(level, name):
        return _open()[f"{tiles.GROUP}/{level}/{name}"]

    def level_for(z):
        """The finest level whose pixel is not smaller than the tile pixel at zoom z."""
        tile_px = 360.0 / (_T * 2 ** z)
        return int(min(tiles.NLEVELS - 1, max(0, math.floor(math.log2(tile_px / grid.RES)))))

    def tile_ll(z, x, y):
        n = 2 ** z
        lat = lambda yy: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
        return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)

    def _centers(z, x, y):
        """Lon and lat of the T x T tile pixel centres (Web Mercator)."""
        W, S, E, N = tile_ll(z, x, y)
        lon = W + (np.arange(_T) + 0.5) * (E - W) / _T
        n = 2 ** z
        world = 2 * math.pi * _R
        tpx = world / (n * _T)
        ty1 = world / 2 - y * world / n
        ys = ty1 - (np.arange(_T) + 0.5) * tpx
        lat = np.degrees(np.arctan(np.sinh(ys / _R)))
        return lon, lat, (W, S, E, N)

    def _sample(z, x, y, year, bands, lv=None):
        """Nearest-neighbour sample of `bands` at the tile pixels from level `lv`
        (the level for z unless given) -> (k, T, T) uint16 arrays, 0 outside
        the grid; None if no overlap."""
        lon, lat, (W, S, E, N) = _centers(z, x, y)
        if E <= grid.WEST or W >= grid.EAST or N <= grid.SOUTH or S >= grid.NORTH:
            return None
        lv = level_for(z) if lv is None else lv
        res = grid.RES * 2 ** lv
        H = -(-grid.HEIGHT // 2 ** lv)
        W_ = -(-grid.WIDTH // 2 ** lv)
        cols = np.floor((lon - grid.WEST) / res).astype(np.int64)
        rows = np.floor((grid.NORTH - lat) / res).astype(np.int64)
        okc, okr = (cols >= 0) & (cols < W_), (rows >= 0) & (rows < H)
        if not okc.any() or not okr.any():
            return None
        c0, c1 = int(cols[okc].min()), int(cols[okc].max()) + 1
        r0, r1 = int(rows[okr].min()), int(rows[okr].max()) + 1
        t = year - tiles.YEARS[0]
        rr = np.clip(rows - r0, 0, r1 - r0 - 1)[:, None]
        cc = np.clip(cols - c0, 0, c1 - c0 - 1)[None, :]
        ok = okr[:, None] & okc[None, :]
        # the bands' blocks side by side through the cache: each block is a
        # chunk fetch over http, about 0.3 s of round trip when not cached
        out = []
        for a in regions(lv, [(b, t, r0, r1, c0, c1) for b in bands]):
            a = a[rr, cc]
            a[~ok] = 0
            out.append(a)
        return np.stack(out)

    def _encode(rgba):
        buf = _io.BytesIO()
        _Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    # Rendered PNGs, keyed by everything that goes into one, so a year seen
    # once costs nothing the second time and a warmed year is already here
    # when the slider reaches it. The bucket has no edge cache (Cloudflare
    # answers DYNAMIC), so this is the only cache between a chunk and the
    # screen. About 100 KB a tile; PNG_CACHE tiles, oldest out first.
    _png = _OD()
    tiles.PNG_CACHE = 1200

    def _remember(key, png):
        _png[key] = png
        _png.move_to_end(key)
        while len(_png) > tiles.PNG_CACHE:
            _png.popitem(last=False)
        return png

    def render(z, x, y, year, mode="tc", gain=1.0, coarse=0):
        """PNG bytes or None (outside the grid, no data, or outside MIN_Z..MAX_Z).
        `coarse` draws from that many pyramid levels above the one for z: a
        quarter of the chunks per level, a softer picture. Cached."""
        if z < tiles.MIN_Z or z > tiles.MAX_Z or year not in tiles.YEARS or mode not in tiles.MODES:
            return None
        coarse = int(max(0, coarse))
        key = (z, x, y, year, mode, round(float(gain), 3), coarse)
        hit = _png.get(key, _MISS)
        if hit is not _MISS:
            _png.move_to_end(key)
            return hit
        lv = min(tiles.NLEVELS - 1, level_for(z) + coarse)
        raw = _sample(z, x, y, year, tiles.MODES[mode], lv)
        if raw is None:
            return _remember(key, None)
        inside = _inside(z, x, y)
        valid = (raw > 0).all(0) & inside
        if not valid.any():
            return _remember(key, None)
        refl = raw.astype(np.float32) * tiles.SCALE + tiles.OFFSET
        out = np.zeros((_T, _T, 4), np.uint8)
        if mode == "ndvi":
            nir, red = refl[0], refl[1]
            den = np.where(valid, nir + red, 1.0)
            v = np.where(valid & (den != 0), (nir - red) / den, tiles.NDVI_LO)
            idx = np.clip((v - tiles.NDVI_LO) / (tiles.NDVI_HI - tiles.NDVI_LO) * 255, 0, 255).astype(np.uint8)
            out[..., :3] = tiles.NDVI_LUT[idx]
        else:
            rgb = np.clip(refl / tiles.REFL_MAX * gain, 0, 1) ** (1.0 / tiles.GAMMA) * 255
            out[..., :3] = np.moveaxis(rgb, 0, -1).astype(np.uint8)
        out[..., 3] = np.where(valid, 255, 0)
        return _remember(key, _encode(out))

    tiles.set_clip, tiles.array, tiles.level_for, tiles.render = set_clip, array, level_for, render
    return (tiles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # New England Landsat mosaic, beside CTrees biomass

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/kentstephen/ne-landsat-ctrees-marimo/blob/main/ne-landsat-ctrees-pair.py)

    A side-by-side view of 26 years of forest change across the six New
    England states, 2000 to 2025. Two maps share one camera: pan or zoom
    either and both move; hover a hexagon on either side and its outline is
    drawn on both.

    **Left**: the annual leaf-on Landsat mosaic, one year at a time, in true
    colour or NDVI. The picture is never covered by a fill.
    **Right**: CTrees aboveground biomass folded to H3 hexagons over the same
    view, showing either the change between two years or the stock at one,
    with the per-pixel uncertainty drawn as opacity.

    Nothing is read from disk. Every layer streams over HTTP or S3 from
    public buckets, so the notebook runs anywhere with a network connection.

    ## Datasets

    | Layer | Source | Resolution and extent | Licence |
    |---|---|---|---|
    | Leaf-on Landsat mosaic | [landsat-mosaics-new-england](https://source.coop/kentstephen/landsat-mosaics-new-england) on Source Cooperative, built by [ne-landsat-temporal-mosaic](https://github.com/kentstephen/ne-landsat-temporal-mosaic) from Landsat Collection 2 surface reflectance | Zarr v3 multiscales pyramid, 30 m to 3840 m, annual 2000 to 2025, with a `source` plane recording which acquisition each pixel came from | CC0-1.0 |
    | Aboveground biomass | CTrees global AGB, Icechunk store on AWS Open Data (`ctrees-agb-100m-global`). Yang, Saatchi et al. 2026, doi 10.82924/7vmb-zv66 | 100 m, annual 2000 to 2025, Mg/ha, with a residual standard error per pixel and year | CC-BY 4.0 |
    | State clip | TIGER/Line 2024, U.S. Census Bureau, the six states dissolved to the 3 nautical mile limit. Shipped as `supplemental/tiger_states.parquet` beside the mosaic | Vector | Public domain |
    | Water mask | National Hydrography Dataset High Resolution, USGS: lakes, ponds, reservoirs, estuaries, bays and wide rivers of 1 ha and up. Shipped as `supplemental/nhd_water_bodies.parquet` | Vector | Public domain |
    | Wildlands | Wildlands of New England GIS Data 1900-2022, Harvard Forest Data Archive HF435. Foster, Johnson and Hall 2023. Shipped as `supplemental/hf435_wildlands.parquet` | 426 polygons | CC0 |
    | State and county boundaries | Overture Maps divisions, PMTiles from [cboettig/overturemaps](https://source.coop/cboettig/overturemaps) on Source Cooperative, release 2026-02-18.0 | Vector tiles, read live by the browser | CDLA Permissive 2.0 |
    | Town under a click | Overture Maps divisions, GeoParquet from [fused/overture](https://source.coop/fused/overture) on Source Cooperative, release 2026-05-20-0, one DuckDB point query | Vector | CDLA Permissive 2.0 |
    | Basemap and place search | Tiles by OpenFreeMap, geocoding by Photon (komoot), both on OpenStreetMap data | | ODbL |

    ## How the panes are built

    - **The mosaic** is rendered in the kernel from the pyramid. At each zoom
      the level whose pixel is the finest not smaller than the screen pixel
      is read, so a zoomed-out view costs a few chunks and a zoomed-in one
      reads level 0 at 30 m.
    - **The biomass** is read for the box on screen once, all 26 years, then
      averaged into H3 hexagons at resolution 9 and coarser. The hexagons
      fold from zoom 9 and stop at resolution 9, about one CTrees pixel,
      while the picture keeps zooming. Below zoom 9 the right pane is empty.
    - **Clipping** happens in the kernel. The picture goes transparent
      outside the six states; the fold averages only the CTrees samples
      inside them and drops a hexagon when fewer than half its samples are
      inside. Water is removed the same way, at the pixel level.
    - **The loss year** is the first year the biomass falls past both a
      fraction and an absolute threshold from the window's running maximum.
      It has no fill of its own: the year slider shows the fall itself, and
      a click names the year for the cell.

    ## Controls

    - **YEAR** (the slider, `[` `]`, or the arrow keys): 2000 to 2025, one
      year of the mosaic at a time. `b` blinks between the first and last
      year.
    - **SHOW** (`n`): true colour, or NDVI on the fixed -0.1 to 0.9 scale.
      **SCALE** is a gain on true colour, under a gamma of 2.
    - **FIND**: a place name. Enter or a click flies both maps there.
    - **FILL** (keys `1` and `2`): **change**, the biomass at the window's
      to-end minus the from-end, blue where it rose and orange where it
      fell, faint where the change is smaller than the uncertainty of its
      two ends; **stock**, the biomass at the to-end, on greens, faint
      where the uncertainty exceeds it.
    - **WINDOW** (the slider, keys `-` `=` for the from end and `_` `+` for
      the to end): whole years 2000 to 2025. A window change is a frame,
      never a fetch.
    - **WILDLANDS** (`w`): the 426 Wildlands of New England, conserved land
      left to natural process. Boundaries only, gold on both panes. A click
      inside one names it in the readout.
    - **ADMIN** (`s`): state and county boundaries, charcoal on both panes.
      The basemap's own admin lines are hidden.
    - **WATER MASK** (`m`, on by default): open water leaves the CTrees fold
      at the pixel level, so no hexagon is drawn over water and a shore
      hexagon averages its land pixels only. Off, water reads as zero
      biomass.
    - `L` toggles the basemap labels, `F` full screen.

    ## A click

    Click a hexagon for its story. Under the left pane: the cell's mosaic
    NDVI for every year 2000 to 2025, with the year on the slider marked.
    Under the right pane: the CTrees stock at both window ends, the change
    in Mg/ha and in Mg across the cell's hectares, the loss year, the
    uncertainty, the 26-year biomass series, the town, county and state the
    click fell in, and the wildland, if any. Both series share the same 26
    years, so the two charts line up.
    """)
    return


@app.cell
def _(SUPP, grid, tiles):
    # ---- constants ----------------------------------------------------------
    # the picture: our own pyramid, one year at a time
    PIC_YEARS = tuple(tiles.YEARS)
    YEAR0 = 2000
    SCALE0 = 1.0   # the gamma in tiles.render carries the lift now; this is a trim
    PIC_MODES = (("tc", "true colour"), ("ndvi", "NDVI"))
    # the AOI: the build grid's bbox. The CTrees read box is clamped to it,
    # and the picture's TileLayer takes it as its extent.
    AOI = (grid.WEST, grid.SOUTH, grid.EAST, grid.NORTH)

    # the biomass window: the whole run by default (the story is movement)
    CT_YEARS = tuple(range(2000, 2026))
    WIN_FROM0, WIN_TO0 = 2000, 2025

    # The zoom -> H3 ladder: BASE_RES at ZOOM0, one step finer every PER_RES
    # zoom units, clamped, then coarsened until the view's expected cell
    # count fits CELL_BUDGET. Capped at res 9, one hexagon per CTrees pixel;
    # the picture keeps zooming.
    ZOOM0, PER_RES, BASE_RES = 6.2, 1.4, 6
    MIN_RES, MAX_RES = 5, 9
    CELL_BUDGET = 300_000

    # ---- CTrees AGB: one Icechunk store, 26 annual slices, 100 m 4326 -------
    # int16 stored, /10 for Mg/ha, nodata -9999; `uncertainty` the residual
    # standard error, same shape and scale. Chunks (1, 2000, 2000). Pixel
    # edges from (-180, 90); centres half a pixel in.
    CT_BUCKET = "ctrees-agb-100m-global"
    CT_PREFIX = "agb_100m_global"
    CT_GROUP = "aboveground_biomass"
    CT_RES = 360.0 / 405000
    CT_X0, CT_Y0 = -180.0, 90.0
    CT_NODATA = -9999
    # samples per year the fold reads: a stride fits the window to this
    CT_MAX_PX = 3_000_000
    CT_READERS = 8
    # Icechunk ships with num_bytes_chunks = 0, so nothing is cached and every
    # pan refetches from S3 (measured 2026-09-06: a warm read cost the same
    # 0.35 s as a cold one). With a budget it is 0.01 s. New England is a fixed
    # 5 x 5 chunk block per year per variable, 650 chunks for all 26 years of
    # agb, so 1.5 GB holds most of a session's working set.
    # The first read stays what it is: 208 MB at about 35 MB/s, and raising
    # max_concurrent_requests past 8 does not help (8/32/128 -> 6.4/12.6/8.4 s).
    CT_CACHE_BYTES = 1_500_000_000
    CT_CACHE_REFS = 5_000_000
    # the loss year: the first year the drop from the window's running
    # maximum passes both of these (a smoothed product ramps; a threshold on
    # the cumulative drop reads better than one on a single step)
    LOSS_FRAC = 0.25
    LOSS_ABS = 10.0

    # ---- the clip: supplemental/tiger_states.parquet ------------------------
    # the six states dissolved, raw TIGER to the 3 nmi limit. Both panes are
    # clipped to it in the kernel; see the header.
    STATE_PATH = f"{SUPP}/tiger_states.parquet"

    # ---- the wildlands: supplemental/hf435_wildlands.parquet -----------------------
    # 426 polygons, 1.5 MB, the whole of New England. Small enough to cross to
    # the browser once at build, so the toggle is browser-side only.
    # Boundaries, never filled; off until asked for.
    WILD_PATH = f"{SUPP}/hf435_wildlands.parquet"
    # gold, one line, alpha 200. Gold is bright against the greens and
    # the dark mosaic, and it stays distinct from the orange loss end of
    # DIV_RAMP by luminance. The cost of alpha below 255 is the one logged
    # earlier: a translucent line takes some of its cast from what is under
    # it, the mosaic on the left pane and the hex fills on the right, so the
    # same paint reads a little differently across the two panes.
    WILD_LINE = (255, 199, 44, 200)
    WILD_WIDTH = 1.4

    # ---- the water: supplemental/nhd_water_bodies.parquet -------------------------------
    # NHD HR lakes, ponds, reservoirs, estuaries, bays and wide rivers, 1 ha
    # and up (water.py in the source repo). Rasterized per read box beside the
    # state clip and taken out of the fold at the pixel level. Never drawn.
    WATER_PATH = f"{SUPP}/nhd_water_bodies.parquet"
    # ---- admin: Overture Maps divisions, live from Source Cooperative -------
    # Two repositories, each for what it holds. cboettig/overturemaps
    # (release 2026-02-18.0) has regions and counties as PMTiles: the browser
    # range-reads the tiles it needs, draws the lines, and answers a click's
    # state and county from the tiles it already has. Nothing crosses the
    # kernel. fused/overture (release 2026-05-20-0) has the whole divisions
    # theme as geo-partitioned GeoParquet, localities included, which the
    # PMTiles are not: the town is one DuckDB point query against it with the
    # bbox column pushed down, 9 s the first time (79 footers), 1 to 2 s after.
    ADMIN_PM = "https://data.source.coop/cboettig/overturemaps/2026-02-18.0"
    ADMIN_PQ = "s3://fused/overture/2026-05-20-0/theme=divisions/type=division_area/*.parquet"
    ADMIN_STATES = ("US-CT", "US-MA", "US-ME", "US-NH", "US-RI", "US-VT")
    # states charcoal and opaque, counties the same ink at half strength and
    # thinner, so the two levels read as one family at two weights
    ADMIN_LINE = (35, 35, 40, 255)
    ADMIN_WIDTH = 1.2
    ADMIN_COUNTY_LINE = (35, 35, 40, 130)
    ADMIN_COUNTY_WIDTH = 0.8

    # the hex rings, hover and picked. One flat line each, drawn by MapLibre
    # rather than deck: deck has no shader-side AA on paths and the overlay is
    # interleaved, so a thin deck line stair-steps in MapLibre's context
    # whatever the MSAA setting, and the white hover ring over the dark mosaic
    # showed it as black notches.
    # This was briefly a casing, a dark line under a light one, because no
    # single colour clears all three hex fills: each ramp runs light to dark,
    # so a flat ring bottoms out somewhere, and white died on DIV_RAMP's quiet
    # #f2f2f2. It fixed the contrast and read cheap, two
    # stacked lines rather than one, so it is gone. The right pane needs no
    # ring at all now: it lifts the hovered cell's own fill instead, which is
    # what the contrast argument was ever about.
    # So the hover ring is left-pane only, and answers to the mosaic alone:
    # near-white, high alpha, because true colour over New England is dark
    # forest. It goes quiet at the pale top of the NDVI ramp, EMRLD's #d3f2a3,
    # which is most of the forest in NDVI mode; the way out there would be to
    # pick the colour from picMode.
    RING_HOVER = (245, 245, 245, 230)
    RING_PICK = (255, 200, 40, 200)
    RING_W = 1.2
    RING_PICK_W = 1.8

    # the hover mark on the RIGHT pane: no ring at all, the cell's own fill
    # lifted. H3HexagonLayer draws it, so it is the same polygon shader as the
    # fills and there is no path to alias. deck's own highlightColor would be
    # one blend for every cell, and every fixed blend has a null: white dies on
    # DIV_RAMP's #f2f2f2 middle, black dies on the dark end of greens. So the
    # lift is computed per cell instead, from that cell's colour: a pale cell
    # darkens, a dark cell lightens, always by HL_LIFT. Nothing is ever null,
    # and the mark is a whole hexagon rather than a 1.2px line, so it needs far
    # less contrast than a ring did to read.
    # The cost: the hovered cell's colour moves, so for the moment you hover it
    # sits a step off its true place on the ramp. The story panel prints the
    # number, and the step is small enough to keep the ordering.
    HL_LIFT = 0.22
    # luminance (0-1) above which a cell darkens instead of lightening
    HL_MID = 0.55
    # for a cell with no data, nothing to lift: a flat wash instead
    HL_FLAT = (120, 120, 120, 130)

    VIEW_W, VIEW_H = 700, 760
    STRIP_MINIMAL = True
    PAD = 1.3
    SETTLE = 0.35
    HEX_ZOOM = 9.0
    # An override only: named here, the deck layers go before this style
    # layer. Empty means the widget finds the first text layer in whatever
    # basemap style loaded, which survives a change of basemap.
    LABELS_SLOT = ""
    RASTER_TILE = 256
    # home: Katahdin Woods and Waters National Monument, ME-05, 76,633 acres,
    # made a monument in 2016 and the largest wildland designated inside the
    # mosaic era. Centred on the property's centroid at a zoom that holds the
    # whole of it with a little context, and past HEX_ZOOM so both panes are
    # live on the first draw.
    HOME = {"longitude": -68.72, "latitude": 45.96, "zoom": 9.5}

    FILLS = ("change", "stock")
    FILL_NAMES = {
        "change": "biomass at the window's to-end minus the from-end (Mg/ha): blue rose, orange fell; faint where smaller than the uncertainty",
        "stock": "biomass at the window's to-end (Mg/ha); faint where the uncertainty exceeds it",
    }
    FILL_SHORT = {"change": "change", "stock": "stock"}
    ALPHA_FILL = 235
    ALPHA_QUIET = 70
    # gain blue, loss orange, quiet white: the axis a protanope keeps
    DIV_RAMP = ("#1f5fa8", "#5b8fd0", "#a9c5e8", "#f2f2f2", "#f4b46b", "#e07b1c", "#a85200")
    # the stock: matplotlib Greens less its white end (greens are fine)
    GREENS = ("#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#006d2c", "#00441b")
    return (
        ALPHA_FILL,
        ALPHA_QUIET,
        AOI,
        BASE_RES,
        CELL_BUDGET,
        CT_BUCKET,
        CT_CACHE_BYTES,
        CT_CACHE_REFS,
        CT_GROUP,
        CT_MAX_PX,
        CT_NODATA,
        CT_PREFIX,
        CT_READERS,
        CT_RES,
        CT_X0,
        CT_Y0,
        CT_YEARS,
        DIV_RAMP,
        FILLS,
        FILL_NAMES,
        FILL_SHORT,
        GREENS,
        HEX_ZOOM,
        HL_FLAT,
        HL_LIFT,
        HL_MID,
        HOME,
        LABELS_SLOT,
        LOSS_ABS,
        LOSS_FRAC,
        MAX_RES,
        MIN_RES,
        PAD,
        PER_RES,
        PIC_MODES,
        PIC_YEARS,
        RASTER_TILE,
        RING_HOVER,
        RING_PICK,
        RING_PICK_W,
        RING_W,
        SCALE0,
        SETTLE,
        ADMIN_COUNTY_LINE,
        ADMIN_COUNTY_WIDTH,
        ADMIN_LINE,
        ADMIN_PM,
        ADMIN_PQ,
        ADMIN_STATES,
        ADMIN_WIDTH,
        STATE_PATH,
        STRIP_MINIMAL,
        VIEW_H,
        VIEW_W,
        WATER_PATH,
        WILD_LINE,
        WILD_PATH,
        WILD_WIDTH,
        WIN_FROM0,
        WIN_TO0,
        YEAR0,
        ZOOM0,
    )


@app.cell
def _(
    BASE_RES,
    CELL_BUDGET,
    MAX_RES,
    MIN_RES,
    PAD,
    PER_RES,
    VIEW_H,
    VIEW_W,
    ZOOM0,
    math,
):
    # ---- the camera -> box and res --------------------------------------------
    CELL_KM2 = {5: 252.9, 6: 36.13, 7: 5.161, 8: 0.7373, 9: 0.1053, 10: 0.01505, 11: 0.00215, 12: 0.000307}

    def _lat_to_y(lat):
        r = math.radians(lat)
        return (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2

    def _y_to_lat(y):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y))))

    def view_to_bbox(vs):
        """The flat camera footprint (W, S, E, N) of ONE pane; the widget reports
        the pane's canvas size (`w`, `h`) with every move."""
        world = 512 * (2 ** vs["zoom"])
        w, h = vs.get("w") or VIEW_W, vs.get("h") or VIEW_H
        half_lon = 360.0 * w / world / 2
        yc, half_y = _lat_to_y(vs["latitude"]), h / world / 2
        return (
            vs["longitude"] - half_lon,
            _y_to_lat(yc + half_y),
            vs["longitude"] + half_lon,
            _y_to_lat(yc - half_y),
        )

    def pad_box(b, f=PAD):
        dx, dy = (b[2] - b[0]) * (f - 1) / 2, (b[3] - b[1]) * (f - 1) / 2
        return (max(-179.9, b[0] - dx), max(-85.0, b[1] - dy), min(179.9, b[2] + dx), min(85.0, b[3] + dy))

    def clip_box(b, aoi):
        """The box inside the AOI, or None where they do not meet."""
        out = (max(b[0], aoi[0]), max(b[1], aoi[1]), min(b[2], aoi[2]), min(b[3], aoi[3]))
        return out if out[2] > out[0] and out[3] > out[1] else None

    def box_km2(b):
        w = (b[2] - b[0]) * 111.32 * math.cos(math.radians((b[1] + b[3]) / 2))
        return abs(w * (b[3] - b[1]) * 110.57)

    def res_for_view(vs, box):
        r = max(MIN_RES, min(MAX_RES, BASE_RES + math.floor((vs["zoom"] - ZOOM0) / PER_RES)))
        while r > MIN_RES and box_km2(box) / CELL_KM2[r] > CELL_BUDGET:
            r -= 1
        return r

    def contains(outer, inner):
        return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]

    return CELL_KM2, clip_box, contains, pad_box, res_for_view, view_to_bbox


@app.cell
def _(XarrayContext, coordinates_to_cells, pa, udf):
    # THE FOLD IS THE H3 UDF INSIDE DATAFUSION. One context, every fold.
    ctx = XarrayContext()
    ctx.register_udf(
        udf(
            lambda la, lo, r: pa.array(coordinates_to_cells(la.to_numpy(), lo.to_numpy(), r[0].as_py())),
            [pa.float64(), pa.float64(), pa.int32()],
            pa.uint64(),
            "stable",
            name="h3_latlng_to_cell",
        )
    )
    return (ctx,)


@app.cell
def _(
    CT_BUCKET,
    CT_CACHE_BYTES,
    CT_CACHE_REFS,
    CT_GROUP,
    CT_MAX_PX,
    CT_NODATA,
    CT_PREFIX,
    CT_READERS,
    CT_RES,
    CT_X0,
    CT_Y0,
    CT_YEARS,
    STATE_GEOM,
    WATER_GEOMS,
    asyncio,
    ctx,
    icechunk,
    math,
    np,
    time,
    xr,
):
    # ---- CTrees: the Icechunk store, one fold per (box, res), all 26 years ------
    # The grid is plate carree, so a lon/lat box IS a window. The fold reads
    # every year of `agb` and `uncertainty` under the box on a stride s so the
    # samples fit CT_MAX_PX per year (an unbiased sample of the cell's mean),
    # then ONE DataFusion query with 26 means and the two uncertainty ends.
    # A window change is a frame; the fetch happens once per (box, res).
    # The caller clamps the box to the AOI: nothing outside New England is read.
    # Inside the box, samples outside the six states (STATE_GEOM) are flagged
    # (the polygon rasterised onto the window's sample grid, rasterio, plate
    # carree, cached with the window) and the fold averages the inside ones
    # only; a cell with fewer than half its samples inside is dropped.
    # every field is set: naming one leaves the rest at 0, which is what the
    # "cache that can only keep 0 references" warning is about
    _cfg = icechunk.RepositoryConfig.default()
    _cfg.caching = icechunk.CachingConfig(
        num_snapshot_nodes=10_000,
        num_chunk_refs=CT_CACHE_REFS,
        num_transaction_changes=100,
        num_bytes_attributes=10_000_000,
        num_bytes_chunks=CT_CACHE_BYTES,
    )
    _st = icechunk.s3_storage(bucket=CT_BUCKET, prefix=CT_PREFIX, region="us-west-2", anonymous=True)
    _repo = icechunk.Repository.open(_st, config=_cfg)
    _ds = xr.open_zarr(_repo.readonly_session(branch="main").store, zarr_format=3, group=CT_GROUP, chunks=None, mask_and_scale=False)
    _ti = {int(y): i for i, y in enumerate(_ds.time.dt.year.values)}
    _win = {}
    _inside = {}
    # the water polygons in an STRtree, once, for the per-box bbox query
    from shapely import STRtree as _STRtree, box as _box
    _water_tree = _STRtree(WATER_GEOMS) if len(WATER_GEOMS) else None
    _sem = asyncio.Semaphore(CT_READERS)
    _fold_lock = asyncio.Lock()

    def _window_ix(box):
        W_, S_, E_, N_ = box
        H, W = _ds.agb.shape[1], _ds.agb.shape[2]
        c0, c1 = max(0, int(math.floor((W_ - CT_X0) / CT_RES))), min(W, int(math.ceil((E_ - CT_X0) / CT_RES)))
        r0, r1 = max(0, int(math.floor((CT_Y0 - N_) / CT_RES))), min(H, int(math.ceil((CT_Y0 - S_) / CT_RES)))
        return c0, c1, r0, r1

    # Reads are serialised at about 0.25 s per chunk whatever the threading, so
    # the biomass comes as ONE call over the time axis and the uncertainty only
    # for the two window ends, cached per year.
    async def _read(var, tsel, r0, r1, c0, c1, stride):
        loop = asyncio.get_running_loop()
        async with _sem:
            return await loop.run_in_executor(
                None, lambda: np.asarray(_ds[var].isel(time=tsel, y=slice(r0, r1, stride), x=slice(c0, c1, stride)).values)
            )

    async def ct_window(box, stride, years):
        """(A (26, h, w) int16 biomass, U {year: (h, w) int16 uncertainty for
        the `years` asked}, lon of the columns, lat of the rows) under the
        box, every `stride`-th pixel; or None."""
        c0, c1, r0, r1 = _window_ix(box)
        if c1 <= c0 or r1 <= r0:
            return None
        key = (stride, r0, r1, c0, c1)
        got = _win.get(key)
        if got is None:
            A = (await _read("agb", slice(None), r0, r1, c0, c1, stride)).astype(np.int16)
            got = (A, {})
            _win[key] = got
            if len(_win) > 6:
                _win.pop(next(iter(_win)))
        A, U = got
        need = [y for y in years if y not in U]
        if need:
            parts = await asyncio.gather(*(_read("uncertainty", _ti[y], r0, r1, c0, c1, stride) for y in need))
            for y, u in zip(need, parts):
                U[y] = u.astype(np.int16)
        lon = CT_X0 + (c0 + stride * np.arange(A.shape[2]) + 0.5) * CT_RES
        lat = CT_Y0 - (r0 + stride * np.arange(A.shape[1]) + 0.5) * CT_RES
        M = _inside.get(key)
        if M is None:
            from rasterio import features
            from rasterio.transform import from_origin

            tr_ = from_origin(CT_X0 + c0 * CT_RES, CT_Y0 - r0 * CT_RES, CT_RES * stride, CT_RES * stride)
            shape = A.shape[1:]
            m_state = features.rasterize([(STATE_GEOM, 1)], out_shape=shape, transform=tr_, fill=0, dtype="uint8").astype(bool)
            # the water: only the polygons whose bbox touches the read box, so
            # a town-sized box does not rasterize every pond in New England
            hit = _water_tree.query(_box(CT_X0 + c0 * CT_RES, CT_Y0 - r1 * CT_RES, CT_X0 + c1 * CT_RES, CT_Y0 - r0 * CT_RES)) if _water_tree is not None else []
            m_water = (features.rasterize([(WATER_GEOMS[i], 1) for i in hit], out_shape=shape, transform=tr_, fill=0, dtype="uint8").astype(bool)
                       if len(hit) else np.zeros(shape, bool))
            M = (m_state, m_water)
            _inside[key] = M
            if len(_inside) > 6:
                _inside.pop(next(iter(_inside)))
        return A, U, lon, lat, M

    # means over the samples inside the states only; a cell is kept when at
    # least half its samples are inside, so a hexagon straddling the coast or
    # the NY line is drawn when it is mostly New England and dropped when it
    # is mostly not (the picture under it stops at the line itself).
    _AGG = ", ".join(f"avg(CASE WHEN inside = 1 AND a_{y} >= 0 THEN CAST(a_{y} AS DOUBLE) / 10.0 END) AS a_{y}" for y in CT_YEARS)
    INSIDE_MIN = 0.5

    async def ct_fold(box, res, y0, y1, water=True):
        """Per res cell over the box: the sample count (`npx`), how many were
        valid in the last year (`nok`), the mean biomass per year in Mg/ha
        (`a_2000`..`a_2025`, null where no sample was valid) and the mean
        uncertainty at the window's two ends (`u0`, `u1`). `inside` is the
        states less the water when `water`, the states alone when not.
        (table or None, stats)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        c0, c1, r0, r1 = _window_ix(box)
        if c1 <= c0 or r1 <= r0:
            return None, "CTrees: nothing under the view"
        full = (c1 - c0) * (r1 - r0)
        stride = max(1, int(math.ceil(math.sqrt(full / CT_MAX_PX))))
        got = await ct_window(box, stride, (y0, y1))
        if got is None:
            return None, "CTrees: nothing under the view"
        A, U, lon, lat, (m_state, m_water) = got
        M = (m_state & ~m_water) if water else m_state
        tr = time.time()
        h, w = A.shape[1], A.shape[2]
        LON, LAT = np.meshgrid(lon, lat)
        last = CT_YEARS[-1]
        data = {f"a_{y}": (("y", "x"), A[i]) for i, y in enumerate(CT_YEARS)}
        data |= {"u0": (("y", "x"), U[y0]), "u1": (("y", "x"), U[y1])}
        data |= {"lat": (("y", "x"), LAT), "lon": (("y", "x"), LON)}
        data |= {"inside": (("y", "x"), M.astype(np.uint8))}
        async with _fold_lock:
            try:
                ctx.deregister_table("ct")
            except Exception:
                pass
            ctx.from_dataset("ct", xr.Dataset(data, coords={"y": np.arange(h), "x": np.arange(w)}), chunks={"y": 256})
            out = ctx.sql(f"""
                SELECT * FROM (
                  SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell,
                         count(*) AS npx,
                         sum(CASE WHEN inside = 1 THEN 1 ELSE 0 END) AS nin,
                         sum(CASE WHEN inside = 1 AND a_{last} >= 0 THEN 1 ELSE 0 END) AS nok,
                         {_AGG},
                         avg(CASE WHEN inside = 1 AND u0 >= 0 THEN CAST(u0 AS DOUBLE) / 10.0 END) AS u0,
                         avg(CASE WHEN inside = 1 AND u1 >= 0 THEN CAST(u1 AS DOUBLE) / 10.0 END) AS u1
                  FROM ct
                  WHERE lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
                    AND a_{last} > {CT_NODATA}
                  GROUP BY cell
                ) WHERE nin >= {INSIDE_MIN} * npx AND nok > 0
            """).to_arrow_table()
        return out, (
            f"CTrees {w:,}x{h:,} samples x 26 years (stride {stride}, {int(100 * stride)} m) read {tr - t0:.1f} s · fold {out.num_rows:,} {time.time() - tr:.1f} s ({y0}, {y1} uncertainty)"
            + (f" · water masked {int((m_state & m_water).sum()):,} px" if water else " · water in")
            + ("" if len(WATER_GEOMS) else " · no water layer on the bucket (supplemental/nhd_water_bodies.parquet)")
        )

    return (ct_fold,)


@app.cell
def _(STATE_PATH, WATER_PATH, read_gpq):
    # ---- the clip polygon and the water polygons: read once ---------------------
    _sb = read_gpq(STATE_PATH)
    STATE_GEOM = _sb[_sb.kind == "new_england"].geometry.iloc[0]
    # tens of thousands of polygons; geometry only, the attributes never matter
    # here. Missing on the bucket: no water is masked and the fold says so.
    try:
        WATER_GEOMS = list(read_gpq(WATER_PATH, columns=["geometry"]).geometry.values)
    except OSError:
        WATER_GEOMS = []
    return STATE_GEOM, WATER_GEOMS


@app.cell
def _(STATE_GEOM, asyncio, grid, np, tiles, time):
    # ---- the picture: our pyramid, one PNG per (z, x, y, year, mode) ------------
    # tiles.render does the read and the colour; the store stays open between
    # calls and `refresh` re-opens it, which is what a running rebuild needs.
    # Pixels outside the six states go transparent (whole tiles answer from
    # their box; only edge tiles run the point test, once each).
    tiles.set_clip(STATE_GEOM)
    _stat = {"served": 0, "blank": 0, "ms": 0.0}
    _gain = {"v": 1.0}

    def pic_set_scale(v):
        v = float(min(4.0, max(0.1, v)))
        if v == _gain["v"]:
            return False
        _gain["v"] = v
        return True

    def pic_stats():
        return dict(_stat, scale=_gain["v"])

    async def pic_tile_png(z, x, y, year, mode="tc", coarse=0, dropped=None):
        """PNG bytes for Web Mercator tile (z, x, y) of the year's mosaic, or
        None (outside the grid, outside the pyramid's zooms, or no data).
        `coarse` levels above the exact one, for a sweep through the years.
        `dropped()` true when nobody waits for it any more: not rendered."""
        gain = _gain["v"] if mode == "tc" else 1.0
        t0 = time.time()

        def work():
            if dropped is not None and dropped():
                return None
            return tiles.render(z, x, y, year, mode, gain, coarse)

        _live["n"] += 1
        try:
            png = await asyncio.to_thread(work)
        finally:
            _live["n"] -= 1
        if dropped is not None and dropped():
            return None
        _stat["served" if png is not None else "blank"] += 1
        _stat["ms"] += 1000 * (time.time() - t0)
        return png

    # tiles the widget is waiting on right now; the warm yields to them
    _live = {"n": 0}

    # ---- warming: the years ahead of a sweep, into the PNG cache ------------
    # The widget names the view's tiles and the years in its direction of
    # travel, nearest first. Each year is rendered WARM_AT tiles at a time,
    # coarse, into tiles.render's cache; a new warm cancels the one before,
    # so a reversal or a rest never queues stale years behind live tiles.
    WARM_AT = 6
    _warm = {"task": None, "sem": None}

    async def pic_warm(z, xys, years, mode="tc", coarse=1, on_year=None):
        t = _warm["task"]
        if t is not None and not t.done():
            t.cancel()
        if _warm["sem"] is None:
            _warm["sem"] = asyncio.Semaphore(WARM_AT)
        gain = _gain["v"] if mode == "tc" else 1.0

        async def one(yr, x, y):
            async with _warm["sem"]:
                while _live["n"] > 0:
                    await asyncio.sleep(0.05)
                await asyncio.to_thread(tiles.render, z, x, y, yr, mode, gain, coarse)

        async def run():
            for yr in years:
                await asyncio.gather(*(one(yr, x, y) for x, y in xys), return_exceptions=True)
                _stat["warmed"] = _stat.get("warmed", 0) + len(xys)
                if on_year:
                    on_year(yr, _stat["warmed"])

        _warm["task"] = asyncio.get_running_loop().create_task(run())

    # ---- the NDVI series: one clicked cell, every mosaic year ------------------
    # The pyramid is local zarr, so this is a disk read: measured 0.01-0.03 s.
    # The cell footprint is approximated by a square of its own area centred on
    # the click, sampled from the finest level whose pixel is at most a quarter
    # of that side. tiles.YEARS is 2000-2025, the same 26 years as CTrees, so
    # this chart and the biomass sparkline share an x axis.
    def ndvi_series(lon, lat, side_m):
        """(mean NDVI per mosaic year as float32 with NaN where no pixel was
        valid, the pyramid level, how many pixels), or (None, 0, 0)."""
        import math

        lv = 0
        while lv + 1 < tiles.NLEVELS and grid.RES * 2 ** (lv + 1) * 111_000 <= side_m / 4:
            lv += 1
        px = grid.RES * 2 ** lv
        rows = -(-grid.HEIGHT // 2 ** lv)
        cols = -(-grid.WIDTH // 2 ** lv)
        half = max(1, int(round(side_m / 111_000 / px / 2)))
        c = int((lon - grid.WEST) / px)
        r = int((grid.NORTH - lat) / px)
        c0, c1 = max(0, c - half), min(cols, c + half + 1)
        r0, r1 = max(0, r - half), min(rows, r + half + 1)
        if c1 <= c0 or r1 <= r0:
            return None, lv, 0
        nt = len(tiles.YEARS)
        got = tiles.regions(lv, [(b, t, r0, r1, c0, c1) for b in ("nir", "red") for t in range(nt)])
        nir = np.stack(got[:nt]).astype(np.float32)
        red = np.stack(got[nt:]).astype(np.float32)
        ok = (nir > 0) & (red > 0)
        n = nir * tiles.SCALE + tiles.OFFSET
        d = red * tiles.SCALE + tiles.OFFSET
        den = n + d
        v = np.where(ok & (den != 0), (n - d) / np.where(den == 0, 1.0, den), np.nan)
        with np.errstate(invalid="ignore"):
            out = np.nanmean(v.reshape(v.shape[0], -1), 1).astype(np.float32)
        return out, lv, (r1 - r0) * (c1 - c0)

    # ---- the source series: which look each of the cell's pixels came from --
    # The pyramid's `source` plane, level 0 only, classes each 30 m pixel of a
    # year: 0 own-year Landsat 5/8/9, 1 the year before, 2 the year after,
    # 3 own-year Landsat 7, 4 nodata, 5 unmatched, 255 not computed (the
    # tile-year is not in the ladder). The cell is the same square as the
    # NDVI series, on level 0 whatever the hexagon size: a res 8 cell is
    # about 30 x 30 pixels a year, 26 years, through the block cache side
    # by side.
    SRC_CLASSES = (0, 1, 2, 3, 4, 5, 255)

    def source_series(lon, lat, side_m):
        """(26 x 7 counts over SRC_CLASSES, pixels per year) or (None, 0)."""
        px = grid.RES
        half = max(1, int(round(side_m / 111_000 / px / 2)))
        c = int((lon - grid.WEST) / px)
        r = int((grid.NORTH - lat) / px)
        c0, c1 = max(0, c - half), min(grid.WIDTH, c + half + 1)
        r0, r1 = max(0, r - half), min(grid.HEIGHT, r + half + 1)
        if c1 <= c0 or r1 <= r0:
            return None, 0
        nt = len(tiles.YEARS)
        got = tiles.regions(0, [("source", t, r0, r1, c0, c1) for t in range(nt)])
        counts = np.zeros((nt, len(SRC_CLASSES)), np.int64)
        for t, a in enumerate(got):
            for k, cls in enumerate(SRC_CLASSES):
                counts[t, k] = int((a == cls).sum())
        return counts, (r1 - r0) * (c1 - c0)

    PIC_YEARS_T = tuple(int(y) for y in tiles.YEARS)

    # the NDVI ramp and the zoom range the legend and the TileLayer need
    NDVI_HEX, NDVI_LO, NDVI_HI = tiles.NDVI_HEX, tiles.NDVI_LO, tiles.NDVI_HI
    MIN_Z, MAX_Z = tiles.MIN_Z, tiles.MAX_Z
    return (
        MAX_Z,
        MIN_Z,
        NDVI_HEX,
        NDVI_HI,
        NDVI_LO,
        PIC_YEARS_T,
        ndvi_series,
        pic_set_scale,
        pic_stats,
        pic_tile_png,
        pic_warm,
        source_series,
    )


@app.cell
def _(WILD_PATH, np, read_gpq):
    # ---- the wildlands: every ring, once ------------------------------------
    # one float32 lon/lat run and the vertex index each ring starts at: a
    # compact way across the wire, unpacked to GeoJSON once. 426 polygons and
    # about 105,000 vertices, read whole at build and never again.
    def rings_of(geoms):
        """(float32 lon/lat pairs, uint32 ring starts, how many polygons)."""
        xs, starts, at = [], [0], 0
        for geom in geoms:
            if geom is None or geom.is_empty:
                continue
            polys = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
            for poly in polys:
                for ring in [poly.exterior, *poly.interiors]:
                    c = np.asarray(ring.coords, np.float32)
                    if len(c) < 2:
                        continue
                    xs.append(c)
                    at += len(c)
                    starts.append(at)
        coords = np.concatenate(xs, 0) if xs else np.zeros((0, 2), np.float32)
        return (np.ascontiguousarray(coords, np.float32),
                np.asarray(starts, np.uint32), len(geoms))

    def wild_rings():
        return rings_of(read_gpq(WILD_PATH, columns=["geometry"]).geometry.values)

    # the click: which wildland, if any, the point falls in. 426 polygons, so
    # the whole frame with its attributes is held once and the sindex answers
    # in microseconds. Reported whether or not the boundaries are drawn.
    _wl = {}

    def wild_at(lon, lat):
        """The wildland containing the point as a dict, or None."""
        import geopandas as gpd
        from shapely.geometry import Point

        if not _wl:
            _wl["g"] = read_gpq(WILD_PATH)
        g = _wl["g"]
        pt = Point(lon, lat)
        hits = g.iloc[g.sindex.query(pt, predicate="within")]
        if not len(hits):
            return None
        # nested polygons happen (a reserve inside a park): the smallest wins
        r = hits.loc[hits.AcresGIS.idxmin()]
        return {"name": r.PropName, "owner": r.FeeOwner, "state": r.State,
                "acres": float(r.AcresGIS), "year": int(r.YearOrig)}

    return wild_at, wild_rings


@app.cell
def _(ADMIN_PQ, duckdb):
    # ---- the town: one point query against fused/overture, live ---------------
    # Overture's division_area as Fused geo-partitions it on Source Cooperative:
    # 79 GeoParquet files, 6.3 GB, each row with a bbox struct. DuckDB reads
    # the footers, keeps the row groups whose bbox stats can hold the point,
    # and runs ST_Contains on what is left. Its own connection, a cursor per
    # call so a click and the warm-up can overlap, and the object cache on so
    # the footers are read once: the first call is about 9 s, the rest 1 to 2.
    import threading as _th

    _dv = {"con": None, "err": None}
    _lock = _th.Lock()

    def _connect():
        with _lock:
            if _dv["con"] is None and _dv["err"] is None:
                try:
                    c = duckdb.connect()
                    for ext in ("spatial", "httpfs"):
                        try:
                            c.execute(f"LOAD {ext}")
                        except Exception:
                            c.execute(f"INSTALL {ext}; LOAD {ext}")
                    # the bucket is public; DuckDB still wants a secret to
                    # sign with, so it gets an empty one for this endpoint
                    c.execute(
                        "CREATE SECRET IF NOT EXISTS source_coop (TYPE s3, PROVIDER config, KEY_ID '', SECRET '', "
                        "REGION 'us-west-2', ENDPOINT 'data.source.coop', URL_STYLE 'path', USE_SSL true); "
                        "SET enable_object_cache=true;"
                    )
                    _dv["con"] = c
                except Exception as e:
                    _dv["err"] = e
            return _dv["con"]

    _Q = (
        "SELECT subtype, names.primary AS name, region "
        f"FROM read_parquet('{ADMIN_PQ}', hive_partitioning=0) "
        "WHERE bbox.xmin <= $x AND bbox.xmax >= $x AND bbox.ymin <= $y AND bbox.ymax >= $y "
        "AND country = 'US' AND class = 'land' AND subtype IN ('region', 'county', 'locality') "
        "AND ST_Contains(geometry, ST_Point($x, $y))"
    )

    def division_at(lon, lat):
        """{town, county, state, state_code} for the point, any of them None.
        Raises on a failed read so the caller can say so."""
        c = _connect()
        if c is None:
            raise _dv["err"]
        rows = c.cursor().execute(_Q, {"x": float(lon), "y": float(lat)}).fetchall()
        out = {"town": None, "county": None, "state": None, "state_code": None}
        for sub, name, region in rows:
            if sub == "locality" and out["town"] is None:
                out["town"] = name
            elif sub == "county" and out["county"] is None:
                out["county"] = name
            elif sub == "region" and out["state"] is None:
                out["state"], out["state_code"] = name, (region or "").split("-")[-1] or None
        return out

    # the footers, read now rather than on the first click
    def _warm():
        try:
            division_at(-68.72, 45.96)
        except Exception:
            pass

    _th.Thread(target=_warm, daemon=True).start()
    return (division_at,)


@app.cell
def _(duckdb):
    # ---- DuckDB: the click row and the tables under the map -------------------
    con = duckdb.connect()
    return (con,)


@app.cell
def _(
    ALPHA_FILL,
    ALPHA_QUIET,
    CT_YEARS,
    DIV_RAMP,
    GREENS,
    LOSS_ABS,
    LOSS_FRAC,
    np,
    pa,
):
    # ---- a FRAME: the window's two ends, the change, the loss year, two fills ---
    def _ramp(stops):
        st = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in stops], np.float64)
        r = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(st)), st[:, k]) for k in range(3)], 1)
        return r.round().astype(np.uint8)

    DIV, GRN = _ramp(DIV_RAMP), _ramp(GREENS)
    _hex = lambda R: ["#%02x%02x%02x" % tuple(int(v) for v in R[i]) for i in range(0, 256, 17)]
    DIV_HEX, GRN_HEX = _hex(DIV), _hex(GRN)
    _YI = {y: i for i, y in enumerate(CT_YEARS)}

    def build_frame(tab, y0, y1):
        """From the fold's per-year means: the stock at the to-end, the change
        between the ends, the loss year (the first year the drop from the
        running maximum since y0 passes LOSS_FRAC of it and LOSS_ABS), the
        deepest such drop, what came back since the window's minimum, and the
        uncertainty at both ends. Two fills and their legends."""
        tab = tab.sort_by("cell")
        n = tab.num_rows
        A = np.stack([tab[f"a_{y}"].to_numpy(zero_copy_only=False).astype(np.float32) for y in CT_YEARS], 0) if n else np.zeros((len(CT_YEARS), 0), np.float32)
        u0 = tab["u0"].to_numpy(zero_copy_only=False).astype(np.float32) if n else np.zeros(0, np.float32)
        u1 = tab["u1"].to_numpy(zero_copy_only=False).astype(np.float32) if n else np.zeros(0, np.float32)
        i0, i1 = _YI[y0], _YI[y1]
        a0, a1 = A[i0], A[i1]
        change = a1 - a0
        z = np.abs(change) / np.sqrt(np.maximum(u0 ** 2 + u1 ** 2, 1e-6))
        # the loss year: the running maximum from y0, the drop from it
        win = A[i0:i1 + 1]
        runmax = np.maximum.accumulate(np.nan_to_num(win, nan=-1.0), axis=0)
        drop = runmax - np.nan_to_num(win, nan=np.inf)
        passed = (drop >= LOSS_ABS) & (drop >= LOSS_FRAC * runmax) & np.isfinite(win)
        passed[0] = False
        any_loss = passed.any(0)
        first = passed.argmax(0)
        lossyear = np.where(any_loss, y0 + first, -1).astype(np.int64)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lost = np.where(np.isfinite(win).any(0), np.nanmax(np.where(np.isfinite(win), drop, 0), axis=0), np.nan).astype(np.float32) if n else np.zeros(0, np.float32)
            wmin = np.nanmin(win, axis=0) if n else np.zeros(0, np.float32)
            ymin = np.where(np.isfinite(win).any(0), y0 + np.nanargmin(np.where(np.isfinite(win), win, np.inf), axis=0), -1) if n else np.zeros(0, np.int64)
        recovered = (a1 - wmin).astype(np.float32)
        ushare = np.where(a1 > 0, u1 / np.maximum(a1, 1e-6), np.inf).astype(np.float32)
        scored = np.isfinite(change)
        cells = pa.table({
            "cell": tab["cell"],
            "npx": tab["npx"],
            "stock": pa.array(a1.astype(np.float32)),
            "stock0": pa.array(a0.astype(np.float32)),
            "change": pa.array(change.astype(np.float32)),
            "z": pa.array(z.astype(np.float32)),
            "lossyear": pa.array(lossyear.astype(np.int16)),
            "lost": pa.array(lost),
            "recovered": pa.array(recovered),
            "ymin": pa.array(np.asarray(ymin).astype(np.int16)),
            "unc0": pa.array(u0.astype(np.float32)),
            "unc1": pa.array(u1.astype(np.float32)),
            "ushare": pa.array(ushare),
        })
        cellid = tab["cell"].to_numpy().astype(np.uint64)
        ok = change[scored]
        if len(ok) >= 2:
            p2, p98 = (float(q) for q in np.percentile(ok, [2, 98]))
            lim = max(abs(p2), abs(p98), 1.0)
        else:
            lim = 1.0
        t = np.clip((np.where(scored, change, 0) / lim + 1) / 2, 0, 1)
        # loss is the orange side: the ramp runs blue (gain) .. white .. orange (loss)
        rgb_change = DIV[((1 - t) * 255).round().astype(np.int64)]
        a_change = np.where(scored, ALPHA_QUIET + (ALPHA_FILL - ALPHA_QUIET) * np.clip(z, 0, 1), ALPHA_QUIET).round().astype(np.int64)
        st = a1[np.isfinite(a1)]
        s_hi = max(float(np.percentile(st, 98)), 1.0) if len(st) >= 2 else 1.0
        ts = np.clip(np.nan_to_num(a1, nan=0.0) / s_hi, 0, 1)
        rgb_stock = GRN[(ts * 255).round().astype(np.int64)]
        a_stock = np.where(np.isfinite(a1) & (ushare <= 1.0), ALPHA_FILL, ALPHA_QUIET)

        def fill(kind, hit=None):
            c, a = (rgb_change, a_change) if kind == "change" else (rgb_stock, a_stock)
            return np.ascontiguousarray(np.concatenate([c, a[:, None].astype(np.uint8)], axis=1)).astype(np.uint8)

        def legend(kind):
            if kind == "change":
                return [{"ramp": list(reversed(DIV_HEX)), "lo": f"{y0} to {y1}: rose {lim:.0f} Mg/ha", "hi": f"fell {lim:.0f}",
                         "title": "biomass at the to-end minus the from-end, symmetric about zero on this view's p2/p98; faint where the change is smaller than the uncertainty of its two ends"}]
            return [{"ramp": GRN_HEX, "lo": f"{y1} stock 0", "hi": f"{s_hi:.0f} Mg/ha",
                     "title": "mean biomass at the to-end, stretched to this view's p98; faint where the uncertainty exceeds the stock"}]

        n_loss = int(any_loss.sum())
        n_sig = int((scored & (z >= 1)).sum())
        med = float(np.nanmedian(change)) if scored.any() else float("nan")
        score = (
            f"CTrees {y0} to {y1}: {n:,} cells · median change {med:+.1f} Mg/ha · {n_sig:,} changed past their uncertainty · {n_loss:,} fell past the threshold"
        )
        return {"cells": cells, "cellid": cellid, "A": A, "y0": y0, "y1": y1, "lim": lim, "s_hi": s_hi,
                "fill": fill, "legend": legend, "score": score, "n_loss": n_loss}

    return (build_frame,)


@app.cell
def _(anywidget, asyncio, traitlets):
    class PairMap(anywidget.AnyWidget):
        """Two maplibre maps in a row, one camera. LEFT: our mosaic as tiles
        the kernel renders from the pyramid (custom messages, PNG bytes back),
        keyed by year and mode.
        RIGHT: an H3HexagonLayer (highPrecision) from cell ids + rgba. Hover on
        either pane: h3-js cell at the frame's res, its ring drawn on BOTH.
        The wildland and state boundaries are MapLibre line layers on each
        pane, never filled, built once from the binary rings sent at build.

        Kernel -> browser: `cells` (uint64 LE), `colors` (rgba u8), `wild_xy`
        (float32 lon/lat) + `wild_idx` (uint32 ring starts, both set once),
        `config` (JSON),
        `status` / `panel` (right, the CTrees story) / `panel_l` (left, the
        NDVI series) / `legend` (strings for the strip), custom `tile`
        replies.
        Browser -> kernel: `view` (JSON lon/lat/zoom + the pane's w/h on every
        moveend), `pick` (JSON: the clicked cell as hex, or null, with the
        state and county under the click from the admin tiles), `ctl`
        (JSON: year, mode, scale, the window, fill, labels, water, refresh)."""

        cells = traitlets.Bytes(b"").tag(sync=True)
        colors = traitlets.Bytes(b"").tag(sync=True)
        wild_xy = traitlets.Bytes(b"").tag(sync=True)
        wild_idx = traitlets.Bytes(b"").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        panel = traitlets.Unicode("").tag(sync=True)
        panel_l = traitlets.Unicode("").tag(sync=True)
        legend = traitlets.Unicode("[]").tag(sync=True)
        view = traitlets.Unicode("").tag(sync=True)
        pick = traitlets.Unicode("").tag(sync=True)
        ctl = traitlets.Unicode("").tag(sync=True)

        def __init__(self, **kw):
            super().__init__(**kw)
            self.tile_fn = None
            self.warm_fn = None
            self._dropped = set()  # tile ids the widget stopped waiting for
            self.on_msg(self._on_custom)

        def _on_custom(self, widget, content, buffers):
            if not isinstance(content, dict):
                return
            kind = content.get("kind")
            if kind == "warm":
                if self.warm_fn is None:
                    return
                try:
                    c = content
                    asyncio.get_running_loop().create_task(self.warm_fn(
                        int(c["z"]), [(int(x), int(y)) for x, y in c.get("tiles", [])],
                        [int(y) for y in c.get("years", [])], c.get("mode", "tc"), int(c.get("coarse", 1))))
                except Exception:
                    pass
                return
            if kind == "drop":
                # deck aborts a tile the moment its layer or view goes; the
                # render it asked for is skipped if it has not started yet,
                # so a sweep does not queue every passed year's tiles ahead
                # of the year the slider rests on
                self._dropped.add(content.get("id"))
                if len(self._dropped) > 4096:
                    self._dropped = set(list(self._dropped)[-1024:])
                return
            if kind != "tile":
                return
            try:
                asyncio.get_running_loop().create_task(self._tile(content))
            except RuntimeError as e:
                self.send({"kind": "tile", "id": content.get("id"), "err": f"no loop: {e}"})

        async def _tile(self, c):
            if self.tile_fn is None:
                self.send({"kind": "tile", "id": c["id"], "err": "no tile_fn (re-run the wiring cell)"})
                return
            tid = c["id"]
            try:
                png = await self.tile_fn(int(c["z"]), int(c["x"]), int(c["y"]), int(c["year"]), c.get("mode", "tc"), int(c.get("coarse", 0)),
                                         lambda: tid in self._dropped)
                if tid in self._dropped:
                    self._dropped.discard(tid)
                    return
            except Exception as e:
                self.send({"kind": "tile", "id": c["id"], "err": f"{type(e).__name__}: {e}"})
                return
            if png is None:
                self.send({"kind": "tile", "id": c["id"], "empty": True})
            else:
                self.send({"kind": "tile", "id": c["id"]}, buffers=[png])

        _esm = r"""
        import maplibregl from "https://cdn.jsdelivr.net/npm/maplibre-gl@5.24.0/+esm";
        // deck.gl as its own self-contained dist bundle (one luma.gl inside),
        // with h3-js loaded first as the global `h3` the bundle looks up when it
        // evaluates (its one external). Per-package ESM builds do not hold
        // together from a CDN: esm.sh hangs in its build queue on the @loaders.gl
        // ranges geo-layers pulls in, and jsdelivr +esm resolves
        // @luma.gl/shadertools to two versions, which luma refuses to start.
        // Two static files, one version each.
        const H3_URL = "https://cdn.jsdelivr.net/npm/h3-js@4.5.0/dist/h3-js.umd.js";
        const DECK_URL = "https://cdn.jsdelivr.net/npm/deck.gl@9.3.10/dist.min.js";
        const PMTILES_URL = "https://cdn.jsdelivr.net/npm/pmtiles@4.5.0/dist/pmtiles.js";
        function loadScript(url, ready) {
          if (ready()) return Promise.resolve();
          return new Promise((ok, bad) => {
            let s = document.querySelector(`script[src="${url}"]`);
            if (!s) { s = document.createElement("script"); s.src = url; s.async = false; document.head.appendChild(s); }
            if (ready()) return ok();
            s.addEventListener("load", () => ok(), {once: true});
            s.addEventListener("error", () => bad(new Error("failed to load " + url)), {once: true});
          });
        }
        await loadScript(H3_URL, () => !!globalThis.h3?.cellToBoundary);
        await loadScript(DECK_URL, () => !!globalThis.deck?.MapboxOverlay);
        await loadScript(PMTILES_URL, () => !!globalThis.pmtiles?.Protocol);
        // the admin tiles come straight off the Source Cooperative bucket by
        // range request: one protocol handler, shared by both maps
        maplibregl.addProtocol("pmtiles", new globalThis.pmtiles.Protocol().tile);
        const {MapboxOverlay, BitmapLayer, TileLayer, H3HexagonLayer} = globalThis.deck;
        const {latLngToCell, getResolution, cellToBoundary} = globalThis.h3;

        const STYLE = "https://tiles.openfreemap.org/styles/positron";

        function bytesOf(v) {
          if (!v) return null;
          if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
          if (v instanceof ArrayBuffer) return new Uint8Array(v);
          if (v.buffer) return new Uint8Array(v.buffer, v.byteOffset || 0, v.byteLength);
          return null;
        }
        function copyOf(u8) { return u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength); }

        function render({model, el}) {
          let cfg = {};
          try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
          const css = document.createElement("link");
          css.rel = "stylesheet"; css.href = "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css";
          const font = "font:12px ui-sans-serif,system-ui,sans-serif";
          const mono = "font:11px ui-monospace,Menlo,monospace";
          const root = document.createElement("div");
          root.className = "sp-root";
          root.style.cssText = "position:relative;width:100%;background:#fff;color:#222;" + font;
          const row = document.createElement("div");
          row.style.cssText = "display:flex;gap:4px;width:100%";
          const mkPane = (side) => {
            const pane = document.createElement("div");
            pane.className = "sp-pane sp-" + side;
            pane.style.cssText = "position:relative;flex:1 1 0;min-width:0;height:" + (cfg.height || 720) + "px;background:#f4f2ee";
            const mapEl = document.createElement("div");
            mapEl.className = "sp-map";
            mapEl.style.cssText = "position:absolute;inset:0";
            const head = document.createElement("div");
            head.className = "sp-head";
            head.style.cssText = "position:absolute;left:8px;top:8px;z-index:5;display:flex;flex-direction:column;gap:.3rem;align-items:flex-start;" +
              "max-width:calc(100% - 72px);box-sizing:border-box;background:rgba(255,255,255,.94);color:#1d1d1b;padding:4px 8px;border-radius:6px;" +
              "box-shadow:0 1px 3px rgba(0,0,0,.18);white-space:nowrap;font-variant-numeric:tabular-nums";
            pane.append(mapEl, head);
            return {pane, mapEl, head};
          };
          const L = mkPane("left"), R = mkPane("right");
          row.append(L.pane, R.pane);
          const strip = document.createElement("div");
          strip.style.cssText = "display:flex;flex-direction:column;gap:.25rem;position:relative;padding:.35rem .4rem;background:#fff;color:#222";
          const status = document.createElement("div");
          status.className = "sp-status";
          status.style.cssText = "font:14px ui-sans-serif,system-ui,sans-serif;color:#444;white-space:pre-wrap";
          // two legends, one under each pane: the picture's on the left, the
          // fill's on the right
          const legends = document.createElement("div");
          legends.style.cssText = "display:flex;gap:4px;width:100%";
          const legendL = document.createElement("div");
          legendL.className = "sp-legend-l";
          legendL.style.cssText = "flex:1 1 0;min-width:0;display:flex;flex-wrap:wrap;gap:.3rem .9rem;align-items:center;font-size:14px";
          const legend = document.createElement("div");
          legend.className = "sp-legend";
          legend.style.cssText = "flex:1 1 0;min-width:0;display:flex;flex-wrap:wrap;gap:.3rem .9rem;align-items:center;font-size:14px";
          legends.append(legendL, legend);
          // two panels, one under each pane, on the same two-column grid as
          // the legends: NDVI from our mosaic on the left, the CTrees story on
          // the right, each under the map it came from
          const panels = document.createElement("div");
          panels.style.cssText = "display:flex;gap:4px;width:100%;align-items:flex-start";
          const panelL = document.createElement("div");
          panelL.className = "sp-panel-l";
          panelL.style.cssText = "flex:1 1 0;min-width:0;font-size:14px";
          const panel = document.createElement("div");
          panel.className = "sp-panel";
          panel.style.cssText = "flex:1 1 0;min-width:0;font-size:14px";
          panels.append(panelL, panel);
          strip.append(legends, panels, status);
          status.hidden = !!cfg.minimal;
          root.append(row, strip);
          el.append(css, root);

          // ---- the controls: the pane headers ------------------------------
          const ACCENT = "#2a5db0";
          const btnCss = font + ";padding:.15rem .55rem;border:0;background:transparent;color:#1d1d1b;cursor:pointer;line-height:1.4;font-variant-numeric:tabular-nums";
          const onCss = (b, on) => { b.style.background = on ? ACCENT : "transparent"; b.style.color = on ? "#fff" : "#1d1d1b"; };
          const labCss = "font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#6b6b68";
          let year = cfg.year, fill = cfg.fill || "change", labelsOn = cfg.labels !== false;
          let picMode = cfg.pic_mode || "tc";
          let wildOn = !!cfg.wild, adminOn = !!cfg.admin, waterOn = cfg.water !== false;
          let scale = Number(cfg.scale) || 1;
          let y0 = cfg.win_from, y1 = cfg.win_to;
          const picYears = cfg.pic_years || [];
          const send = (act, extra) => {
            model.set("ctl", JSON.stringify(Object.assign({act, year, scale, fill, y0, y1, labels: labelsOn, mode: picMode, n: Date.now()}, extra || {})));
            model.save_changes();
          };
          const mkGroup = (head, title, values, get, set, act, cls, isOn) => {
            const wrap = document.createElement("span");
            wrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
            const lab = document.createElement("span"); lab.textContent = title; lab.style.cssText = labCss;
            const seg = document.createElement("span");
            seg.style.cssText = "display:inline-flex;border:1px solid rgba(29,29,27,.28);border-radius:5px;overflow:hidden";
            const btns = values.map((v, i) => {
              const b = document.createElement("button"); b.textContent = String(v.label != null ? v.label : v); b.style.cssText = btnCss;
              if (i) b.style.borderLeft = "1px solid rgba(29,29,27,.18)";
              b.className = cls; b.dataset.value = String(v.value != null ? v.value : v); if (v.title) b.title = v.title;
              b.onclick = () => { set(v.value != null ? v.value : v); style(); update(); send(act); };
              seg.appendChild(b); return b;
            });
            wrap.append(lab, seg);
            rowOf(head).appendChild(wrap);
            const style = () => btns.forEach((b) => onCss(b, isOn ? isOn(b) : b.dataset.value === String(get())));
            style();
            return style;
          };
          // a row wraps: each control keeps its own line, and when the pane
          // is narrower than the row (the notebook beside its text, before
          // full screen) the controls fall to the next line instead of
          // hanging off the pane
          const newRow = (head) => { const r = document.createElement("span"); r.style.cssText = "display:flex;flex-wrap:wrap;gap:.3rem .6rem;align-items:center;max-width:100%"; head.appendChild(r); return r; };
          const rowOf = (head) => head.lastElementChild && head.lastElementChild.tagName === "SPAN" ? head.lastElementChild : newRow(head);
          const rowBreak = (head) => { newRow(head); };
          const fills = (cfg.fills || []).map((f) => ({value: f[0], label: f[1], title: f[2]}));
          const styleFill = mkGroup(R.head, "fill", fills, () => fill, (v) => { fill = v; }, "fill", "sp-fill");
          // the biomass window: a stepped two-handle slider, 2000..2025
          rowBreak(R.head);
          const winYears = cfg.win_years || [];
          const sty = document.createElement("style");
          sty.textContent = [
            ".sp-range{position:relative;width:340px;max-width:100%;height:30px}",
            ".sp-range input{position:absolute;left:0;top:0;width:100%;height:22px;margin:0;background:none;pointer-events:none;-webkit-appearance:none;appearance:none}",
            ".sp-range input:focus{outline:none}",
            ".sp-range input::-webkit-slider-runnable-track{background:none;height:22px}",
            ".sp-range input::-moz-range-track{background:none;height:22px}",
            ".sp-range input::-webkit-slider-thumb{pointer-events:auto;-webkit-appearance:none;appearance:none;width:16px;height:16px;margin-top:3px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-range input::-moz-range-thumb{pointer-events:auto;width:12px;height:12px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-range .trk{position:absolute;left:8px;right:8px;top:9px;height:4px;background:rgba(29,29,27,.22);border-radius:2px}",
            ".sp-range .spn{position:absolute;top:9px;height:4px;background:#2a5db0;border-radius:2px}",
            ".sp-range .tks{position:absolute;left:8px;right:8px;top:19px;display:flex;justify-content:space-between;font-size:9px;color:#6b6b68;line-height:1}",
            ".sp-range .tks span{width:0;display:flex;justify-content:center}",
            ".sp-scale{position:relative;width:120px;height:22px}",
            ".sp-scale input{position:absolute;left:0;top:0;width:100%;height:22px;margin:0;background:none;-webkit-appearance:none;appearance:none}",
            ".sp-scale input:focus{outline:none}",
            ".sp-scale input::-webkit-slider-runnable-track{background:none;height:22px}",
            ".sp-scale input::-moz-range-track{background:none;height:22px}",
            ".sp-scale input::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;margin-top:3px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-scale input::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-scale .trk{position:absolute;left:8px;right:8px;top:9px;height:4px;background:rgba(29,29,27,.22);border-radius:2px}",
            ".sp-scale .spn{position:absolute;left:8px;top:9px;height:4px;background:#2a5db0;border-radius:2px}",
            // the picture year: the scale slider's handle, a long track, the
            // range's ticks
            ".sp-year{height:30px;width:340px;max-width:100%}",
            ".sp-year .tks{position:absolute;left:8px;right:8px;top:19px;display:flex;justify-content:space-between;font-size:9px;color:#6b6b68;line-height:1}",
            ".sp-year .tks span{width:0;display:flex;justify-content:center}",
          ].join("\n");
          el.appendChild(sty);
          // a tick label every fifth year (and the ends); the rest empty
          const tickLabel = (y, i, arr) => (y % 5 === 0 || i === 0 || i === arr.length - 1) ? String(y) : "";
          const winWrap = document.createElement("span");
          winWrap.style.cssText = "display:inline-flex;flex-wrap:wrap;align-items:center;gap:.4rem;max-width:100%";
          const winLab = document.createElement("span"); winLab.textContent = "window"; winLab.style.cssText = labCss;
          const rng = document.createElement("span"); rng.className = "sp-range";
          const trk = document.createElement("span"); trk.className = "trk";
          const spn = document.createElement("span"); spn.className = "spn";
          const tks = document.createElement("span"); tks.className = "tks";
          winYears.forEach((y, i, arr) => { const t = document.createElement("span"); const l = document.createElement("i"); l.style.fontStyle = "normal"; l.textContent = tickLabel(y, i, arr); t.appendChild(l); tks.appendChild(t); });
          const mkRange = () => {
            const r = document.createElement("input"); r.type = "range"; r.min = 0; r.max = Math.max(0, winYears.length - 1); r.step = 1;
            r.title = "the biomass window: drag either end (whole years); release to rebuild the frame (no fetch)"; return r;
          };
          const rFrom = mkRange(), rTo = mkRange();
          const winTxt = document.createElement("span");
          winTxt.style.cssText = "font-variant-numeric:tabular-nums;min-width:6.5em";
          rng.append(trk, spn, tks, rFrom, rTo);
          winWrap.append(winLab, rng, winTxt);
          rowOf(R.head).appendChild(winWrap);
          const styleWin = () => {
            const i0 = Math.max(0, winYears.indexOf(y0)), i1 = Math.max(0, winYears.indexOf(y1)), n = Math.max(1, winYears.length - 1);
            rFrom.value = i0; rTo.value = i1;
            rFrom.style.zIndex = i0 === n ? 3 : 2; rTo.style.zIndex = i1 === 0 ? 3 : 2;
            const usable = rng.clientWidth - 16;
            spn.style.left = (8 + usable * i0 / n) + "px"; spn.style.width = (usable * (i1 - i0) / n) + "px";
            winTxt.textContent = y0 + " to " + y1;
          };
          const onDrag = (which) => {
            let a = Number(rFrom.value), b = Number(rTo.value);
            if (a >= b) { if (which === "from") a = b - 1; else b = a + 1; }
            a = Math.max(0, a); b = Math.min(winYears.length - 1, b);
            y0 = winYears[a]; y1 = winYears[b]; styleWin();
          };
          rFrom.addEventListener("input", () => onDrag("from"));
          rTo.addEventListener("input", () => onDrag("to"));
          let winSent = [y0, y1];
          const winRelease = () => { if (y0 !== winSent[0] || y1 !== winSent[1]) { winSent = [y0, y1]; send("win"); } };
          rFrom.addEventListener("change", winRelease);
          rTo.addEventListener("change", winRelease);
          setTimeout(styleWin, 0);
          try { new ResizeObserver(styleWin).observe(rng); } catch (e) {}

          // the picture year: ONE slider over the pyramid's years
          const yrWrap = document.createElement("span");
          yrWrap.style.cssText = "display:inline-flex;flex-wrap:wrap;align-items:center;gap:.4rem;max-width:100%";
          const yrLab = document.createElement("span"); yrLab.textContent = "year"; yrLab.style.cssText = labCss;
          const yr = document.createElement("span"); yr.className = "sp-scale sp-year";
          const yrTrk = document.createElement("span"); yrTrk.className = "trk";
          const yrSpn = document.createElement("span"); yrSpn.className = "spn";
          const yrTks = document.createElement("span"); yrTks.className = "tks";
          picYears.forEach((y, i, arr) => { const t = document.createElement("span"); const l = document.createElement("i"); l.style.fontStyle = "normal"; l.textContent = tickLabel(y, i, arr); t.appendChild(l); yrTks.appendChild(t); });
          const yri = document.createElement("input"); yri.type = "range"; yri.min = 0; yri.max = Math.max(0, picYears.length - 1); yri.step = 1;
          yri.title = "which year of the mosaic is drawn ([ ] or the arrow keys)";
          const yrTxt = document.createElement("span");
          yrTxt.className = "sp-yeartxt";
          yrTxt.style.cssText = "font-variant-numeric:tabular-nums;min-width:4em";
          yr.append(yrTrk, yrSpn, yrTks, yri);
          yrWrap.append(yrLab, yr, yrTxt);
          rowOf(L.head).appendChild(yrWrap);
          const styleYear = () => {
            const i = Math.max(0, picYears.indexOf(year)), n = Math.max(1, picYears.length - 1);
            yri.value = i;
            yrSpn.style.width = Math.max(0, (yr.clientWidth - 16) * i / n) + "px";
            yrTxt.textContent = String(year);
            try { renderLegendL(); } catch (e) {}
            try { ndviMark(); } catch (e) {}
          };
          // The NDVI chart's year marker follows the slider. The svg carries
          // one x per mosaic year and one y (null where no pixel was valid),
          // so this is a move, never a re-render and never a message.
          const ndviMark = () => {
            const cap = panelL.querySelector("#srccap");
            if (cap) { try { const caps = JSON.parse(cap.dataset.caps); const k = picYears.indexOf(year); if (caps[k] != null) cap.textContent = caps[k]; } catch (e) {} }
            const ln = panelL.querySelector("#ndmk"), dt = panelL.querySelector("#nddot");
            if (!ln || !dt) return;
            const svg = ln.ownerSVGElement; if (!svg) return;
            const xsv = JSON.parse(svg.dataset.mkx || "[]"), ysv = JSON.parse(svg.dataset.mky || "[]");
            const i = picYears.indexOf(year);
            if (i < 0 || i >= xsv.length) return;
            ln.setAttribute("x1", xsv[i]); ln.setAttribute("x2", xsv[i]);
            if (ysv[i] == null) { dt.setAttribute("visibility", "hidden"); }
            else { dt.removeAttribute("visibility"); dt.setAttribute("cx", xsv[i]); dt.setAttribute("cy", ysv[i]); }
          };
          let yrSent = year, yrTimer = null;
          const yrRelease = () => { if (yrTimer) { clearTimeout(yrTimer); yrTimer = null; } if (year !== yrSent) { yrSent = year; send("year"); } };
          const stepYear = (d) => { const from = year; year = step(picYears, year, d); noteYear(from, year); styleYear(); update(); yrRelease(); };
          yri.addEventListener("input", () => { const from = year; year = picYears[Number(yri.value)]; noteYear(from, year); styleYear(); update(); if (yrTimer) clearTimeout(yrTimer); yrTimer = setTimeout(yrRelease, 150); });
          yri.addEventListener("change", yrRelease);
          setTimeout(styleYear, 0);
          try { new ResizeObserver(styleYear).observe(yr); } catch (e) {}

          // the picture mode, then the ramp legend under the left pane
          rowBreak(L.head);
          const modes = (cfg.pic_modes || []).map((m) => ({value: m[0], label: m[1]}));
          const styleMode = mkGroup(L.head, "show", modes, () => picMode, (v) => { picMode = v; }, "mode", "sp-mode");
          const MODE_NAME = {};
          modes.forEach((m) => { MODE_NAME[m.value] = m.label; });
          const renderLegendL = () => {
            legendL.replaceChildren();
            const s = document.createElement("span");
            s.style.cssText = "display:inline-flex;align-items:center;gap:.35rem";
            const who = "mosaic " + year;
            if (picMode === "ndvi") {
              const lo = document.createElement("span");
              lo.textContent = who + " · NDVI " + (cfg.ndvi_lo != null ? cfg.ndvi_lo : -0.1);
              lo.style.opacity = ".75";
              const bar = document.createElement("span");
              bar.style.cssText = "display:inline-block;width:11rem;height:12px;border-radius:2px;background:linear-gradient(90deg," + (cfg.ndvi_ramp || []).join(",") + ")";
              const hi = document.createElement("span");
              hi.textContent = String(cfg.ndvi_hi != null ? cfg.ndvi_hi : 0.9);
              hi.style.opacity = ".75";
              s.append(lo, bar, hi);
              s.title = "NDVI = (NIR - red) / (NIR + red) of the scaled reflectance, one fixed scale for every year; dark low, bright high (Carto Emrld)";
            } else {
              const tx = document.createElement("span");
              tx.textContent = who + " · " + (MODE_NAME[picMode] || picMode);
              tx.style.opacity = ".75";
              s.append(tx);
              s.title = "red, green, blue of the scaled reflectance";
            }
            legendL.appendChild(s);
            if (wildOn) {
              const w = document.createElement("span");
              w.style.cssText = "display:inline-flex;align-items:center;gap:.35rem";
              const chip = document.createElement("span");
              chip.style.cssText = "display:inline-block;width:14px;height:0;border-top:2px solid rgba(" + (cfg.wild_line || [255, 199, 44, 255]).join(",") + ")";
              const t = document.createElement("span"); t.textContent = "wildlands"; t.style.opacity = ".75";
              w.append(chip, t);
              w.title = "Wildlands of New England 2022 (Harvard Forest HF435), boundaries only";
              legendL.appendChild(w);
            }
            if (adminOn) {
              const w = document.createElement("span");
              w.style.cssText = "display:inline-flex;align-items:center;gap:.35rem";
              const chip = document.createElement("span");
              chip.style.cssText = "display:inline-block;width:14px;height:0;border-top:2px solid rgba(" + (cfg.admin_line || [35, 35, 40, 255]).join(",") + ")";
              const chip2 = document.createElement("span");
              chip2.style.cssText = "display:inline-block;width:14px;height:0;border-top:1px solid rgba(" + (cfg.admin_county_line || [35, 35, 40, 130]).join(",") + ")";
              const t = document.createElement("span"); t.textContent = "states · counties"; t.style.opacity = ".75";
              w.append(chip, chip2, t);
              w.title = "Overture Maps divisions, PMTiles on Source Cooperative (cboettig/overturemaps)";
              legendL.appendChild(w);
            }
          };

          // the scale: a gain on the two colour modes
          const SC_MIN = 0.2, SC_MAX = 3;
          const scWrap = document.createElement("span");
          scWrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
          const scLab = document.createElement("span"); scLab.textContent = "scale"; scLab.style.cssText = labCss;
          const scr = document.createElement("span"); scr.className = "sp-scale";
          const scTrk = document.createElement("span"); scTrk.className = "trk";
          const scSpn = document.createElement("span"); scSpn.className = "spn";
          const sc = document.createElement("input"); sc.type = "range"; sc.min = SC_MIN; sc.max = SC_MAX; sc.step = 0.1;
          sc.title = "brightness of the two colour modes (double-click for 1.0)";
          const scTxt = document.createElement("span");
          scTxt.style.cssText = "font-variant-numeric:tabular-nums;min-width:2.4em";
          scr.append(scTrk, scSpn, sc);
          scWrap.append(scLab, scr, scTxt);
          rowOf(L.head).appendChild(scWrap);
          const styleSc = () => {
            sc.value = scale;
            scSpn.style.width = Math.max(0, (scr.clientWidth - 16) * (scale - SC_MIN) / (SC_MAX - SC_MIN)) + "px";
            scTxt.textContent = scale.toFixed(1) + "×";
          };
          let scSent = scale, scTimer = null;
          const scRelease = () => { if (scTimer) { clearTimeout(scTimer); scTimer = null; } if (scale !== scSent) { scSent = scale; send("scale"); } };
          const SC_DEBOUNCE_MS = 150;
          sc.addEventListener("input", () => { scale = Number(sc.value); styleSc(); if (scTimer) clearTimeout(scTimer); scTimer = setTimeout(scRelease, SC_DEBOUNCE_MS); });
          sc.addEventListener("change", scRelease);
          scr.addEventListener("dblclick", (e) => { e.preventDefault(); scale = 1; styleSc(); scRelease(); });
          setTimeout(styleSc, 0);
          try { new ResizeObserver(styleSc).observe(scr); } catch (e) {}

          // wildlands, labels, refresh
          rowBreak(L.head);
          const mkBtn = (text, title, on) => {
            const b = document.createElement("button"); b.textContent = text; b.title = title;
            b.style.cssText = btnCss + ";border:1px solid rgba(29,29,27,.28);border-radius:5px";
            onCss(b, on); rowOf(L.head).appendChild(b); return b;
          };
          // the geometry is already in the browser, so this never asks the kernel
          const wildBtn = mkBtn("wildlands", "the " + (cfg.wild_n || 426) + " Wildlands of New England (2022), boundaries only, gold on both panes (w)", wildOn);
          wildBtn.onclick = () => { wildOn = !wildOn; onCss(wildBtn, wildOn); update(); renderLegendL(); };
          const adminBtn = mkBtn("admin", "state and county boundaries, Overture Maps divisions read live from Source Cooperative, on both panes (s)", adminOn);
          adminBtn.onclick = () => { adminOn = !adminOn; onCss(adminBtn, adminOn); update(); renderLegendL(); };
          // the mask lives in the kernel's fold, so this one asks and waits
          const waterBtn = mkBtn("water mask", "leave lakes, ponds, reservoirs, bays and wide rivers (NHD HR, 1 ha and up) out of the CTrees fold: no hexagon over open water, and a shore hexagon averages its land pixels only (m)", waterOn);
          waterBtn.onclick = () => { waterOn = !waterOn; onCss(waterBtn, waterOn); send("water", {water: waterOn}); };
          const labBtn = mkBtn("labels", "basemap labels (L)", labelsOn);
          labBtn.onclick = () => { labelsOn = !labelsOn; onCss(labBtn, labelsOn); labels(labelsOn); send("labels"); };

          // the geocoder: Photon, from the browser, flies both maps
          const PHOTON = "https://photon.komoot.io/api/";
          const gcWrap = document.createElement("span");
          gcWrap.style.cssText = "position:relative;display:inline-flex;align-items:center;gap:.4rem";
          const gcLab = document.createElement("span"); gcLab.textContent = "find"; gcLab.style.cssText = labCss;
          const gc = document.createElement("input");
          gc.type = "search"; gc.placeholder = "a place…"; gc.autocomplete = "off"; gc.spellcheck = false;
          gc.title = "Photon geocoder: type, pick a hit (arrows, Enter, or click) and both maps fly there";
          gc.style.cssText = "width:12rem;" + font + ";padding:.15rem .45rem;border:1px solid rgba(29,29,27,.28);border-radius:5px;background:#fff;color:#1d1d1b";
          const gcList = document.createElement("div");
          gcList.className = "sp-hits";
          gcList.style.cssText = "position:absolute;left:0;top:calc(100% + 4px);z-index:9;display:none;min-width:100%;max-width:26rem;" +
            "background:#fff;color:#1d1d1b;border:1px solid rgba(29,29,27,.28);border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.18);overflow:hidden";
          gcWrap.append(gcLab, gc, gcList);
          rowOf(L.head).appendChild(gcWrap);
          let gcHits = [], gcSel = -1, gcTimer = null, gcSeq = 0;
          const GC_DEBOUNCE_MS = 250, GC_LIMIT = 6;
          const hitName = (f) => {
            const p = f.properties || {};
            const parts = [p.name, p.street && !p.name ? p.street : null, p.city && p.city !== p.name ? p.city : null,
              p.county && p.county !== p.city && p.county !== p.name ? p.county : null, p.state, p.country];
            return parts.filter((x) => x).join(", ");
          };
          const hitKind = (f) => { const p = f.properties || {}; return [p.osm_value, p.type].filter((x) => x && x !== "yes").join(" · "); };
          const gcHide = () => { gcList.style.display = "none"; gcList.replaceChildren(); gcSel = -1; };
          const gcShow = () => {
            gcList.replaceChildren();
            if (!gcHits.length) { gcHide(); return; }
            gcHits.forEach((f, i) => {
              const row2 = document.createElement("div");
              row2.style.cssText = "padding:.3rem .55rem;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.3;" +
                (i === gcSel ? "background:" + ACCENT + ";color:#fff" : "");
              const nm = document.createElement("div"); nm.textContent = hitName(f);
              const kd = document.createElement("div"); kd.textContent = hitKind(f);
              kd.style.cssText = "font-size:11px;opacity:" + (i === gcSel ? ".85" : ".6");
              row2.append(nm, kd);
              row2.onmousedown = (e) => { e.preventDefault(); gcFly(f); };
              row2.onmouseenter = () => { gcSel = i; gcShow(); };
              gcList.appendChild(row2);
            });
            gcList.style.display = "block";
          };
          const gcAsk = async () => {
            const q = gc.value.trim();
            if (q.length < 2) { gcHits = []; gcHide(); return; }
            const seq2 = ++gcSeq;
            const params = new URLSearchParams({q, limit: String(GC_LIMIT), lang: "en"});
            if (mapL) { const c = mapL.getCenter(); params.set("lon", c.lng.toFixed(4)); params.set("lat", c.lat.toFixed(4)); }
            try {
              const r = await fetch(PHOTON + "?" + params.toString());
              const data = await r.json();
              if (seq2 !== gcSeq) return;
              gcHits = (data.features || []).filter((f) => f.geometry && f.geometry.coordinates);
              gcSel = gcHits.length ? 0 : -1;
              gcShow();
            } catch (e) { if (seq2 === gcSeq) say("search: " + e.message); }
          };
          const gcFly = (f) => {
            const [lon, lat] = f.geometry.coordinates;
            const ext = (f.properties || {}).extent;
            const w = (L.mapEl.clientWidth || 700);
            let zoom = 10;
            if (ext && ext.length === 4) {
              const span = Math.max(Math.abs(ext[2] - ext[0]), Math.abs(ext[1] - ext[3]) * 2, 0.01);
              zoom = Math.log2(360 * (w / 512) / span) - 0.3;
            }
            zoom = Math.max(3.5, Math.min(14, zoom));
            gc.value = hitName(f); gcHits = []; gcHide(); gc.blur();
            if (mapL) mapL.flyTo({center: [lon, lat], zoom, duration: 2000});
            say("→ " + hitName(f) + " · zoom " + zoom.toFixed(1));
          };
          gc.addEventListener("input", () => { if (gcTimer) clearTimeout(gcTimer); gcTimer = setTimeout(gcAsk, GC_DEBOUNCE_MS); });
          gc.addEventListener("focus", () => { if (gcHits.length) gcShow(); });
          gc.addEventListener("blur", () => { setTimeout(gcHide, 120); });
          gc.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "ArrowDown" && gcHits.length) { gcSel = (gcSel + 1) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "ArrowUp" && gcHits.length) { gcSel = (gcSel - 1 + gcHits.length) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "Enter") {
              e.preventDefault();
              if (gcHits.length) gcFly(gcHits[Math.max(0, gcSel)]);
              else { if (gcTimer) clearTimeout(gcTimer); gcAsk().then(() => { if (gcHits.length) gcFly(gcHits[0]); else say("no match: " + gc.value.trim()); }); }
            }
            else if (e.key === "Escape") { gcHide(); gc.blur(); }
          });

          // full screen (the shadow-root walk)
          const isFull = () => {
            let fe = document.fullscreenElement;
            while (fe && fe.shadowRoot && fe.shadowRoot.fullscreenElement) fe = fe.shadowRoot.fullscreenElement;
            return fe === root;
          };
          const stripCss = "display:flex;flex-direction:column;gap:.25rem;position:relative;padding:.35rem .4rem;background:#fff;color:#222";
          const paneHeight = () => {
            const full = isFull();
            root.style.position = "relative";
            root.style.height = full ? "100vh" : "";
            root.style.boxSizing = "border-box";
            for (const pn of [L.pane, R.pane]) pn.style.height = full ? "100vh" : (cfg.height || 720) + "px";
            strip.style.cssText = full
              ? stripCss + ";position:absolute;left:0;right:0;bottom:0;z-index:30;background:rgba(255,255,255,.72);backdrop-filter:blur(7px);-webkit-backdrop-filter:blur(7px);max-height:45vh;overflow-y:auto;box-sizing:border-box;box-shadow:0 -1px 4px rgba(0,0,0,.18)"
              : stripCss;
            styleWin(); styleYear();
          };
          const toggleFull = () => {
            if (isFull()) document.exitFullscreen();
            else root.requestFullscreen().catch((e) => say("fullscreen: " + e.message));
          };
          document.addEventListener("fullscreenchange", () => { setTimeout(paneHeight, 30); });
          window.addEventListener("resize", () => { paneHeight(); });
          // Collapse the strip: a caret at its top right, and a caret at the
          // bottom right of the view to bring it back. The restore caret is
          // lifted clear of the basemap attribution so it never sits on the
          // OpenFreeMap / OpenStreetMap credits.
          const capCss = "position:absolute;z-index:31;width:22px;height:22px;padding:0;border:0;border-radius:4px;cursor:pointer;background:rgba(255,255,255,.82);color:#444;font-size:12px;line-height:22px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.22)";
          const capDn = document.createElement("button");
          capDn.type = "button"; capDn.textContent = "\u25be"; capDn.title = "collapse the panel";
          capDn.style.cssText = capCss + ";top:3px;right:3px";
          const capUp = document.createElement("button");
          capUp.type = "button"; capUp.textContent = "\u25b4"; capUp.title = "show the panel";
          capUp.style.cssText = capCss + ";right:8px;bottom:26px";
          capUp.hidden = true;
          const setStrip = (open) => { strip.hidden = !open; capUp.hidden = open; setTimeout(paneHeight, 0); };
          capDn.addEventListener("click", () => setStrip(false));
          capUp.addEventListener("click", () => setStrip(true));
          strip.appendChild(capDn);
          root.appendChild(capUp);

          const hint = document.createElement("div");
          hint.style.cssText = mono + ";opacity:.55;color:#666";
          hint.textContent = "keys: [ ] year · b blink first/last year · n true colour / NDVI · ; ' scale · 1-2 fill · - = window from · _ + window to · w wildlands · s admin · m water mask · L labels · F full screen · click a hexagon for its story";
          strip.appendChild(hint);
          hint.hidden = !!cfg.minimal;
          const step = (arr, cur, d) => { const i = arr.indexOf(cur); return arr[Math.max(0, Math.min(arr.length - 1, (i < 0 ? 0 : i) + d))]; };
          root.tabIndex = 0;
          root.addEventListener("pointerup", (e) => {
            if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
            setTimeout(() => { try { root.focus({preventScroll: true}); } catch (err) {} }, 0);
          });
          root.addEventListener("keydown", (e) => {
            if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
            const k = e.key;
            if (k === "[" || k === "]" || k === "ArrowLeft" || k === "ArrowRight") { stepYear((k === "]" || k === "ArrowRight") ? 1 : -1); }
            else if (k === ";" || k === "'") { scale = Math.round(10 * Math.max(SC_MIN, Math.min(SC_MAX, scale + (k === "'" ? 0.1 : -0.1)))) / 10; styleSc(); scRelease(); }
            else if (k >= "1" && k <= "9") { const f = fills[Number(k) - 1]; if (f) { fill = f.value; styleFill(); send("fill"); } }
            else if (k === "-" || k === "=") { const v = step(winYears, y0, k === "=" ? 1 : -1); if (v < y1) { y0 = v; styleWin(); winRelease(); } }
            else if (k === "_" || k === "+") { const v = step(winYears, y1, k === "+" ? 1 : -1); if (v > y0) { y1 = v; styleWin(); winRelease(); } }
            else if (k === "n" || k === "N") { const i = modes.findIndex((m) => m.value === picMode); picMode = modes[(i + 1 + modes.length) % modes.length].value; styleMode(); renderLegendL(); update(); send("mode"); }
            // b for blink: the first mosaic year against the last, the two
            // ends of the record, without stepping through 24 years to get there
            else if (k === "b" || k === "B") { const a = picYears[0], z = picYears[picYears.length - 1]; if (a != null && z != null) { const from = year; year = (year === z) ? a : z; noteYear(from, year); styleYear(); update(); yrRelease(); } }
            else if (k === "w" || k === "W") { wildBtn.onclick(); }
            else if (k === "s" || k === "S") { adminBtn.onclick(); }
            else if (k === "m" || k === "M") { waterBtn.onclick(); }
            else if (k === "l" || k === "L") { labBtn.onclick(); }
            else if (k === "f" || k === "F") { toggleFull(); }
            else return;
            e.preventDefault();
          });

          const say = (t) => {
            status.textContent = t || "";
            if (cfg.minimal) status.hidden = !/folding|failed|zoom in past|no match|search:|tile|no water layer/.test(t || "");
          };
          const renderLegend = () => {
            legend.replaceChildren();
            let items = [];
            try { items = JSON.parse(model.get("legend") || "[]"); } catch (e) { items = []; }
            for (const it of items) {
              const s = document.createElement("span");
              s.style.cssText = "display:inline-flex;align-items:center;gap:.35rem";
              if (it.ramp) {
                const bar = document.createElement("span");
                bar.style.cssText = "display:inline-block;width:11rem;height:12px;border-radius:2px;background:linear-gradient(90deg," + it.ramp.join(",") + ")";
                const lo = document.createElement("span"); lo.textContent = it.lo; lo.style.opacity = ".75";
                const hi = document.createElement("span"); hi.textContent = it.hi; hi.style.opacity = ".75";
                s.append(lo, bar, hi); s.title = it.title || "";
              } else {
                const chip = document.createElement("span");
                chip.style.cssText = "display:inline-block;width:12px;height:12px;border-radius:2px;background:" + it.hex;
                const t = document.createElement("span"); t.textContent = it.name + (it.pct != null ? " " + it.pct + "%" : "");
                s.append(chip, t);
              }
              legend.appendChild(s);
            }
          };
          model.on("change:status", () => say(model.get("status")));
          model.on("change:panel", () => { panel.innerHTML = model.get("panel") || ""; });
          model.on("change:panel_l", () => {
            panelL.innerHTML = model.get("panel_l") || "";
            try { ndviMark(); } catch (e) {}   // a fresh chart lands on the year on screen
          });
          model.on("change:legend", renderLegend);

          // ---- the data ----------------------------------------------------
          let hexes = [], N = 0, colors = null, res = -1, hexIndex = new Map(), dataObj = null;
          let wild = null, wildGeo = null;
          const raw = {cells: null, colors: null};
          const grab = (k) => {
            try { const u8 = bytesOf(model.get(k)); raw[k] = u8 && u8.length ? copyOf(u8) : null; }
            catch (e) { raw[k] = null; say("grab " + k + ": " + e.message); }
          };
          function loadCells() {
            const buf = raw.cells;
            if (!buf || !buf.byteLength) { hexes = []; N = 0; hexIndex = new Map(); res = -1; return; }
            const ids = new BigUint64Array(buf);
            N = ids.length; hexes = new Array(N); hexIndex = new Map();
            for (let i = 0; i < N; i++) { const h = ids[i].toString(16); hexes[i] = h; hexIndex.set(h, i); }
            try { res = getResolution(hexes[0]); } catch (e) { res = -1; }
          }
          function loadAttrs() {
            const c8 = raw.colors;
            colors = c8 && c8.byteLength === N * 4 ? new Uint8Array(c8) : null;
            dataObj = N && colors ? {length: N} : null;
          }
          // the wildland rings: deck's binary path format, set once at build
          const pathsOf = (kxy, kix) => {
            const xy = bytesOf(model.get(kxy)), ix = bytesOf(model.get(kix));
            if (!xy || !ix || ix.byteLength < 8) return null;
            const coords = new Float32Array(copyOf(xy));
            const starts = new Uint32Array(copyOf(ix));
            return starts.length > 1 ? {length: starts.length - 1, startIndices: starts,
              attributes: {getPath: {value: coords, size: 2}}} : null;
          };
          // the same rings as GeoJSON, once: what a MapLibre source wants
          const geoOf = (p) => {
            if (!p) return null;
            const xy = p.attributes.getPath.value, st = p.startIndices, lines = [];
            for (let i = 0; i + 1 < st.length; i++) {
              const line = [];
              for (let j = st[i]; j < st[i + 1]; j++) line.push([xy[2 * j], xy[2 * j + 1]]);
              if (line.length > 1) lines.push(line);
            }
            return {type: "Feature", geometry: {type: "MultiLineString", coordinates: lines}, properties: {}};
          };
          function loadWild() {
            try { wild = pathsOf("wild_xy", "wild_idx"); wildGeo = geoOf(wild); } catch (e) { wild = null; wildGeo = null; say("wildlands: " + e.message); }
          }

          // ---- the picture tiles: ask the kernel -----------------------------
          const pending = new Map();
          let tseq = 0;
          const tstat = {asked: 0, got: 0, empty: 0, err: 0, abort: 0};
          model.on("msg:custom", (msg, buffers) => {
            if (!msg) return;
            if (msg.kind !== "tile") return;
            const p = pending.get(msg.id);
            if (!p) return;
            pending.delete(msg.id);
            if (msg.err) { tstat.err++; say("tile: " + msg.err); p.reject(new Error(msg.err)); return; }
            if (msg.empty || !buffers || !buffers.length) { tstat.empty++; p.resolve(null); return; }
            const u8 = bytesOf(buffers[0]);
            createImageBitmap(new Blob([u8], {type: "image/png"})).then(
              (b) => { tstat.got++; p.resolve(b); },
              (e) => { tstat.err++; p.reject(e instanceof Error ? e : new Error("decode")); });
          });
          // the tiles the picture layer asked for last, by tile key: the set a
          // warm is sent for. Reset when a new picture layer starts asking.
          const viewTiles = new Map();
          let viewTilesId = null;
          const getTileDataFor = (yr_, mode, coarse, rid) => ({index, signal}) => new Promise((resolve, reject) => {
            const id = ++tseq;
            tstat.asked++;
            pending.set(id, {resolve, reject});
            const tkey = index.z + "/" + index.x + "/" + index.y;
            if (rid) {
              if (viewTilesId !== rid) { viewTiles.clear(); viewTilesId = rid; }
              viewTiles.set(tkey, {x: index.x, y: index.y, z: index.z});
            }
            model.send({kind: "tile", id, year: yr_, mode: mode || "tc", x: index.x, y: index.y, z: index.z, coarse: coarse || 0});
            if (signal) signal.addEventListener("abort", () => {
              pending.delete(id); tstat.abort++;
              model.send({kind: "drop", id});
              if (rid && viewTilesId === rid) viewTiles.delete(tkey);
              const e = new Error("aborted"); e.name = "AbortError"; reject(e);
            });
          });

          // ---- sweeps: coarse first, sharp at rest ---------------------------
          // Two year changes inside SWEEP_MS make a sweep. While it lasts the
          // picture is drawn one pyramid level up (a quarter of the chunks),
          // the kernel is asked to warm the years ahead in the direction of
          // travel (more of them the longer the sweep runs). REST_MS after the
          // last change the resting year is drawn at the exact level. One
          // picture layer at a time: keeping the last year underneath while
          // the next loaded left a patchwork of two years wherever tiles had
          // not landed, so a year is all or blank.
          const SWEEP_MS = cfg.sweep_ms || 800, REST_MS = cfg.rest_ms || 450;
          const sweep = {last: 0, n: 0, dir: 0, on: false, timer: null};
          const warmAhead = () => {
            if (!viewTiles.size || !sweep.dir) return;
            const i = picYears.indexOf(year), k = Math.min(8, 2 * sweep.n), years = [];
            for (let j = 1; j <= k; j++) { const y = picYears[i + j * sweep.dir]; if (y == null) break; years.push(y); }
            if (!years.length) return;
            const z = viewTiles.values().next().value.z;
            const tl = Array.from(viewTiles.values()).filter((t) => t.z === z).map((t) => [t.x, t.y]);
            model.send({kind: "warm", z, tiles: tl, years, mode: picMode, coarse: 1});
          };
          const noteYear = (from, to) => {
            const now = performance.now();
            sweep.n = (now - sweep.last < SWEEP_MS) ? sweep.n + 1 : 1;
            sweep.last = now;
            sweep.dir = to > from ? 1 : (to < from ? -1 : sweep.dir);
            sweep.on = sweep.n >= 2;
            if (sweep.timer) clearTimeout(sweep.timer);
            sweep.timer = setTimeout(() => { sweep.timer = null; sweep.on = false; sweep.n = 0; update(); }, REST_MS);
            if (sweep.on) warmAhead();
          };

          // ---- the layers ---------------------------------------------------
          let mapL = null, mapR = null, ovL = null, ovR = null;
          let hover = null;
          // The deck layers sit under the basemap labels, which means naming a
          // style layer to go before. Layer ids belong to the style, so a
          // hardcoded one breaks the moment the basemap changes: CARTO's
          // watername_ocean is not in OpenFreeMap positron, and deck answers an
          // unknown beforeId by putting its layers at the top of the style,
          // where the opaque mosaic raster buries the rings. So resolve it
          // from the style that actually loaded: the first symbol layer that
          // draws text, whatever it happens to be called. cfg.labels_slot is
          // honoured when the style really has it, and ignored when it does not.
          let slotId = null;
          const slot = () => {
            if (slotId !== null) return slotId || undefined;
            const m = up(mapL) ? mapL : (up(mapR) ? mapR : null);
            if (!m) return undefined;
            const want = cfg.labels_slot;
            if (want && m.getLayer(want)) { slotId = want; return slotId; }
            const st = m.getStyle();
            const hit = ((st && st.layers) || []).find(
              (l) => l.type === "symbol" && l.layout && l.layout["text-field"] !== undefined);
            slotId = hit ? hit.id : false;
            if (!slotId) say("no label layer in the basemap style: the deck layers go on top");
            return slotId || undefined;
          };
          const ring = (h) => {
            try {
              const r = cellToBoundary(h, true);
              if (!r || !r.length) return null;
              const a = r[0], z = r[r.length - 1];
              return (a[0] === z[0] && a[1] === z[1]) ? r : r.concat([a]);
            } catch (e) { return null; }
          };
          // The rings are MapLibre line layers, not deck PathLayers. deck has
          // no shader-side AA on paths and the overlay is interleaved, so a
          // thin deck line stair-steps in this context whatever the MSAA
          // setting; the white hover ring over the dark mosaic showed it as
          // black notches along the line. MapLibre
          // smooths its own lines, so the same 1.2px reads clean.
          // one source and one layer per ring, with constant paint. Not one
          // source with ["get", ...] expressions: line-opacity takes no
          // data-driven value, so that threw inside addLayer and took the
          // whole update with it.
          const RINGS = [
            {key: "hover", color: () => cfg.ring_hover || [245, 245, 245, 230], w: () => cfg.ring_w || 1.2},
            {key: "pick", color: () => cfg.ring_pick || [255, 200, 40, 200], w: () => cfg.ring_pick_w || 1.8},
          ];
          const srcId = (k) => "sel-" + k + "-src", lyrId = (k) => "sel-" + k + "-line";
          const rgbOf = (c) => "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
          const alphaOf = (c) => (c[3] != null ? c[3] : 255) / 255;
          const EMPTY = {type: "FeatureCollection", features: []};
          // Which maps have fired "load". isStyleLoaded() is the wrong gate
          // for adding layers here: the interleaved deck overlay re-inserts
          // its custom layers on every setProps, which marks the style
          // changed until the next frame, so at the moment update() runs it
          // answers false on every call after the first. Anything gated on
          // it never gets added. The load event fires once and stays true.
          const mapsUp = new WeakSet();
          const up = (m) => !!m && mapsUp.has(m);
          // The boundary lines (wildlands, admin) go under the label slot,
          // above the deck layers. The deck overlay inserts its custom
          // layers at that same slot, and not only inside setProps: it does
          // it again from the render loop, after update() has returned, and
          // on the left pane the mosaic raster is opaque, so a line seated
          // once ends up buried. So the seat is checked on every styledata
          // event, and a line is moved only when a custom layer sits above
          // it; a move fires styledata again, and then nothing is out of
          // place, so it settles.
          const seated = [];
          // getLayersOrder, not getStyle().layers: the serialized style
          // leaves the custom layers out, so deck's are only visible here.
          const seat = (m) => {
            if (!up(m)) return;
            const before = slot();
            if (!before) return;
            let ids;
            try { ids = m.getLayersOrder(); } catch (e) { return; }
            let lastCustom = -1;
            ids.forEach((id, i) => { const l = m.getLayer(id); if (l && l.type === "custom") lastCustom = i; });
            if (lastCustom < 0) return;
            for (const id of seated) {
              const i = ids.indexOf(id);
              if (i >= 0 && i < lastCustom) { try { m.moveLayer(id, before); } catch (e) {} }
            }
          };
          const ringsSetup = (m) => {
            if (!up(m)) return;
            for (const r of RINGS) {
              if (m.getSource(srcId(r.key))) continue;
              const c = r.color();
              try {
                m.addSource(srcId(r.key), {type: "geojson", data: EMPTY});
                const lyr = {
                  id: lyrId(r.key), type: "line", source: srcId(r.key),
                  layout: {"line-cap": "round", "line-join": "round"},
                  paint: {"line-color": rgbOf(c), "line-width": r.w(), "line-opacity": alphaOf(c)},
                };
                m.addLayer(lyr);
              } catch (e) { say("ring " + r.key + ": " + ((e && e.message) || e)); }
            }
          };
          // the picked ring is set up after the hover one, so it sits over it
          // where the two meet. withHover is false on the right, which lifts
          // its fill instead.
          const ringsData = (m, withHover) => {
            if (!m) return;
            const put = (k, h) => {
              const src = m.getSource(srcId(k));
              if (!src) return;
              const r = h ? ring(h) : null;
              src.setData(r ? {type: "Feature", geometry: {type: "LineString", coordinates: r}, properties: {}} : EMPTY);
            };
            put("hover", withHover ? hover : null);
            put("pick", cfg.hit);
          };
          // The rings go at the top of the style, not at the label slot the
          // deck layers use. The interleaved overlay re-inserts its own custom
          // layers there on every setProps, so anything sharing that slot ends
          // up under the mosaic raster, which is opaque: the rings vanished
          // Top of the style means they paint over the
          // labels too, which for a selection mark is the right way round.
          const ringsTop = (m) => {
            if (!m) return;
            for (const r of RINGS) {
              if (!m.getLayer(lyrId(r.key))) continue;
              try { m.moveLayer(lyrId(r.key)); } catch (e) {}
            }
          };
          const picSpec = () => {
            const coarse = sweep.on ? 1 : 0;
            const id = "pic-" + year + "-" + picMode + (picMode === "tc" ? "-s" + (cfg.scale_gen || 0) : "") + (coarse ? "-c" + coarse : "");
            return {id, year, mode: picMode, coarse};
          };
          const mkRaster = (spec) => new TileLayer({
            id: spec.id,
            getTileData: getTileDataFor(spec.year, spec.mode, spec.coarse, spec.id),
            onTileError: (e) => { if (!e || e.name !== "AbortError") say("tile: " + ((e && e.message) || e)); },
            tileSize: cfg.tile || 256,
            maxRequests: cfg.tile_requests || 12,
            minZoom: cfg.min_z || 4, maxZoom: cfg.max_z || 13,
            extent: cfg.bbox || null,
            refinementStrategy: "best-available",
            beforeId: slot(),
            renderSubLayers: (p) => {
              if (!p.data) return null;
              const {west, south, east, north} = p.tile.bbox;
              return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]});
            },
          });
          // The wildland boundaries are a MapLibre line layer too, for the
          // same reason as the rings: as a deck PathLayer it came out
          // stair-stepped. MapLibre's geojson source also simplifies the
          // rings per zoom on its own. One source and one layer, added
          // once, toggled by visibility, and re-seated under the label slot
          // on every update, since the interleaved overlay re-inserts the
          // deck layers at that slot and would otherwise bury it under the
          // mosaic.
          const BOUNDS = [
            {key: "wild", on: () => wildOn, geo: () => wildGeo, color: () => cfg.wild_line || [255, 199, 44, 255], w: () => cfg.wild_width || 1.4},
          ];
          const bSrc = (k) => "bound-" + k + "-src", bLyr = (k) => "bound-" + k + "-line";
          const boundsSync = (m) => {
            if (!up(m)) return;
            for (const b of BOUNDS) {
              const g = b.geo();
              if (!g) continue;
              if (!m.getSource(bSrc(b.key))) {
                const c = b.color();
                try {
                  m.addSource(bSrc(b.key), {type: "geojson", data: g, tolerance: 0.35});
                  m.addLayer({
                    id: bLyr(b.key), type: "line", source: bSrc(b.key),
                    layout: {"line-cap": "round", "line-join": "round", visibility: b.on() ? "visible" : "none"},
                    paint: {"line-color": rgbOf(c), "line-width": b.w(), "line-opacity": alphaOf(c)},
                  }, slot());
                  seated.push(bLyr(b.key));
                } catch (e) { console.error("boundary " + b.key, e); say("boundary " + b.key + ": " + ((e && e.message) || e)); continue; }
              }
              try { m.setLayoutProperty(bLyr(b.key), "visibility", b.on() ? "visible" : "none"); } catch (e) {}
            }
          };
          // the hovered cell's own fill, moved one step away from where it
          // sits: pale cells down, dark cells up. One cell, one H3HexagonLayer
          // over the fills, so it is polygons all the way and nothing aliases.
          const liftColor = (i) => {
            const r = colors[4 * i], g = colors[4 * i + 1], b = colors[4 * i + 2], a = colors[4 * i + 3];
            if (a < 40) return cfg.hl_flat || [120, 120, 120, 130];
            const t = cfg.hl_lift != null ? cfg.hl_lift : 0.22;
            const lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
            return lum > (cfg.hl_mid != null ? cfg.hl_mid : 0.55)
              ? [r * (1 - t), g * (1 - t), b * (1 - t), a]
              : [r + (255 - r) * t, g + (255 - g) * t, b + (255 - b) * t, a];
          };
          const liftLayer = (h) => {
            if (!h || !colors) return null;
            const i = hexIndex.get(h);
            if (i === undefined) return null;
            const c = liftColor(i);
            return new H3HexagonLayer({
              id: "lift",
              data: {length: 1},
              getHexagon: () => h,
              getFillColor: () => c,
              updateTriggers: {getHexagon: [h], getFillColor: [h, dataObj]},
              filled: true, stroked: false, extruded: false,
              highPrecision: true, pickable: false, beforeId: slot(),
            });
          };
          // The admin lines: Overture divisions as PMTiles on Source
          // Cooperative, one vector source per level, tiles range-read by the
          // browser as it moves. Two layers per level: a line, toggled by the
          // button, and a fill at zero opacity that is never seen and never
          // toggled, so queryRenderedFeatures can answer which state and
          // county a click fell in from the tiles already on screen. Land
          // rows in the six states only: the maritime rows would draw the
          // 3 nmi limit through the sea, and the NY towns are not the story.
          const ADMIN = [
            {key: "regions", color: () => cfg.admin_line || [35, 35, 40, 255], w: () => cfg.admin_width || 1.2},
            {key: "counties", color: () => cfg.admin_county_line || [35, 35, 40, 130], w: () => cfg.admin_county_width || 0.8},
          ];
          const aSrc = (k) => "admin-" + k + "-src", aLine = (k) => "admin-" + k + "-line", aFill = (k) => "admin-" + k + "-fill";
          const adminFilter = () => ["all", ["==", ["get", "class"], "land"],
            ["in", ["get", "region"], ["literal", cfg.admin_states || ["US-CT", "US-MA", "US-ME", "US-NH", "US-RI", "US-VT"]]]];
          const adminSync = (m) => {
            if (!up(m) || !cfg.admin_pm) return;
            for (const a of ADMIN) {
              if (!m.getSource(aSrc(a.key))) {
                const c = a.color();
                try {
                  m.addSource(aSrc(a.key), {type: "vector", url: "pmtiles://" + cfg.admin_pm + "/" + a.key + ".pmtiles"});
                  m.addLayer({
                    id: aFill(a.key), type: "fill", source: aSrc(a.key), "source-layer": a.key,
                    filter: adminFilter(), paint: {"fill-opacity": 0},
                  }, slot());
                  m.addLayer({
                    id: aLine(a.key), type: "line", source: aSrc(a.key), "source-layer": a.key,
                    filter: adminFilter(),
                    layout: {"line-cap": "round", "line-join": "round", visibility: adminOn ? "visible" : "none"},
                    paint: {"line-color": rgbOf(c), "line-width": a.w(), "line-opacity": alphaOf(c)},
                  }, slot());
                  seated.push(aFill(a.key), aLine(a.key));
                } catch (e) { console.error("admin " + a.key, e); say("admin " + a.key + ": " + ((e && e.message) || e)); continue; }
              }
              try { m.setLayoutProperty(aLine(a.key), "visibility", adminOn ? "visible" : "none"); } catch (e) {}
            }
          };
          // the state and county under a point, from the tiles on screen.
          // Empty when the tiles are not in yet; the kernel's town query
          // fills the same fields then.
          const adminAt = (m, pt) => {
            const out = {};
            const one = (key) => {
              if (!m.getLayer(aFill(key))) return null;
              const fs = m.queryRenderedFeatures(pt, {layers: [aFill(key)]});
              return fs && fs.length ? fs[0].properties : null;
            };
            try {
              const r = one("regions");
              if (r) { out.state = r.name_en || r["names.primary"] || null; out.state_code = (r.region || "").split("-").pop() || null; }
              const c = one("counties");
              if (c) out.county = c.name_en || c["names.primary"] || null;
            } catch (e) {}
            return out;
          };
          const hexZoomOk = () => !!mapR && mapR.getZoom() >= (cfg.hex_zoom || 9);
          function layersLeft() {
            return [mkRaster(picSpec())];
          }
          function layersRight() {
            const out = [];
            if (dataObj && hexZoomOk()) out.push(new H3HexagonLayer({
              id: "hexes",
              data: {length: N},
              getHexagon: (_, {index}) => hexes[index],
              getFillColor: (_, {index}) => [colors[4 * index], colors[4 * index + 1], colors[4 * index + 2], colors[4 * index + 3]],
              updateTriggers: {getFillColor: [dataObj], getHexagon: [dataObj]},
              filled: true, stroked: false, extruded: false,
              highPrecision: true,
              pickable: false,
              beforeId: slot(),
            }));
            // hover is the lifted fill, not a ring: over a flat colormap a
            // thin line is the hardest thing to see and the easiest to make
            // look cheap. The left pane has no fill to lift, so it keeps its
            // ring. Picked keeps the gold ring on both panes, so selection and
            // hover stay different kinds of mark.
            if (dataObj && hexZoomOk()) {
              const hl = liftLayer(hover);
              if (hl) out.push(hl);
            }
            return out;
          }
          function update() {
            if (ovL) ovL.setProps({layers: layersLeft()});
            if (ovR) ovR.setProps({layers: layersRight()});
            ringsSetup(mapL); ringsData(mapL, true); ringsTop(mapL);
            ringsSetup(mapR); ringsData(mapR, false); ringsTop(mapR);
            boundsSync(mapL); boundsSync(mapR);
            adminSync(mapL); adminSync(mapR);
            seat(mapL); seat(mapR);
          }
          function updateHover() { update(); }

          // the basemap's own admin lines (Positron draws admin levels 2 to
          // 6: countries, states, counties) are hidden for good, so the only
          // boundaries on the map are the ones the ADMIN button draws.
          function hideBasemapBoundaries() {
            for (const m of [mapL, mapR]) {
              if (!up(m)) continue;
              const st = m.getStyle();
              if (!st || !st.layers) continue;
              st.layers.forEach((l) => {
                if (l["source-layer"] === "boundary")
                  m.setLayoutProperty(l.id, "visibility", "none");
              });
            }
          }
          function labels(on) {
            for (const m of [mapL, mapR]) {
              if (!up(m)) continue;
              const st = m.getStyle();
              if (!st || !st.layers) continue;
              st.layers.forEach((l) => {
                if (l.layout && l.layout["text-field"] !== undefined)
                  m.setLayoutProperty(l.id, "visibility", on ? "visible" : "none");
              });
            }
          }

          let seq = 0, lastView = "";
          function sendView() {
            if (!mapL) return;
            const c = mapL.getCenter();
            const v = {longitude: c.lng, latitude: c.lat, zoom: mapL.getZoom(), w: L.mapEl.clientWidth, h: L.mapEl.clientHeight};
            const key = JSON.stringify(v);
            if (key === lastView) return;
            lastView = key;
            v.n = ++seq;
            model.set("view", JSON.stringify(v));
            model.save_changes();
            // no "folding…" guess here: the kernel says so itself when it
            // folds, and a guess stuck on screen whenever the kernel's reply
            // was the same status as before
          }

          const cellAt = (lngLat) => {
            if (res < 0) return null;
            try { const h = latLngToCell(lngLat.lat, lngLat.lng, res); return hexIndex.has(h) ? h : null; }
            catch (e) { return null; }
          };

          function boot() {
            const home = cfg.home || {longitude: -71.3, latitude: 44.05, zoom: 9.2};
            // antialias: true asks for an MSAA context. MapLibre defaults it
            // off because its own line shaders smooth themselves, but the deck
            // overlay is interleaved, so deck draws into this same context and
            // its PathLayer has no shader-side AA of its own: without MSAA the
            // hex rings and the wildland lines come out stair-stepped
            const mk = (elm) => new maplibregl.Map({
              container: elm, style: STYLE,
              center: [home.longitude, home.latitude], zoom: home.zoom,
              attributionControl: {compact: false},
              antialias: true,
            });
            mapL = mk(L.mapEl); mapR = mk(R.mapEl);
            mapL.keyboard.disable(); mapR.keyboard.disable();
            mapR.addControl(new maplibregl.FullscreenControl({container: root}), "top-right");
            mapR.addControl(new maplibregl.NavigationControl({showCompass: false}), "top-right");
            ovL = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck L: " + (e && e.message ? e.message : e))});
            ovR = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck R: " + (e && e.message ? e.message : e))});
            mapL.addControl(ovL); mapR.addControl(ovR);
            let syncing = false;
            const follow = (a, b) => () => {
              if (syncing) return;
              syncing = true;
              b.jumpTo({center: a.getCenter(), zoom: a.getZoom(), bearing: a.getBearing(), pitch: a.getPitch()});
              syncing = false;
            };
            mapL.on("move", follow(mapL, mapR));
            mapR.on("move", follow(mapR, mapL));
            let ready = 0;
            const onLoad = () => { ready++; if (ready === 2) { hideBasemapBoundaries(); labels(labelsOn); update(); sendView(); } };
            mapL.once("load", () => { mapsUp.add(mapL); onLoad(); });
            mapR.once("load", () => { mapsUp.add(mapR); onLoad(); });
            mapL.on("styledata", () => seat(mapL));
            mapR.on("styledata", () => seat(mapR));
            mapL.on("moveend", sendView); mapR.on("moveend", sendView);
            mapL.on("zoom", () => update());
            mapR.on("zoom", () => update());
            for (const m of [mapL, mapR]) {
              m.on("mousemove", (e) => {
                const h = cellAt(e.lngLat);
                if (h !== hover) { hover = h; updateHover(); }
              });
              m.on("mouseout", () => { if (hover) { hover = null; updateHover(); } });
              m.on("click", (e) => {
                const h = cellAt(e.lngLat);
                model.set("pick", JSON.stringify({cell: h, lon: e.lngLat.lng, lat: e.lngLat.lat, admin: adminAt(m, e.point), n: ++seq}));
                model.save_changes();
              });
              m.on("error", (ev) => { if (ev && ev.error && ev.error.message) say("map: " + ev.error.message); });
              new ResizeObserver(() => { try { m.resize(); } catch (e) {} }).observe(m.getContainer());
            }
            window.__spTiles = tstat;
            window.__spMaps = () => [mapL, mapR];
            window.__spLayers = () => ({left: layersLeft().map((l) => l.id), right: layersRight().map((l) => l.id), N, res});
          }

          let pendingLoad = null, needCells = false;
          const flush = () => {
            pendingLoad = null;
            try {
              if (needCells) loadCells();
              needCells = false;
              loadAttrs(); update();
            }
            catch (e) { say("load: " + e.message); console.error(e); }
          };
          const reload = () => { needCells = true; if (!pendingLoad) pendingLoad = setTimeout(flush, 0); };
          const reattr = () => { if (!pendingLoad) pendingLoad = setTimeout(flush, 0); };
          model.on("change:cells", () => { grab("cells"); reload(); });
          model.on("change:colors", () => { grab("colors"); reattr(); });
          model.on("change:config", () => {
            const was = cfg;
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            if (Number(cfg.scale) && Number(cfg.scale) !== scale) { scale = Number(cfg.scale); scSent = scale; styleSc(); }
            if (cfg.fill && cfg.fill !== fill) { fill = cfg.fill; styleFill(); }
            if (cfg.water !== undefined && !!cfg.water !== waterOn) { waterOn = !!cfg.water; onCss(waterBtn, waterOn); }
            if (cfg.pic_mode && cfg.pic_mode !== picMode) { picMode = cfg.pic_mode; styleMode(); }
            if (cfg.labels !== was.labels) { labelsOn = cfg.labels !== false; labels(labelsOn); }
            styleYear(); renderLegendL();
            update();
          });
          try { grab("cells"); grab("colors"); loadCells(); loadAttrs(); loadWild(); renderLegend(); renderLegendL(); say(model.get("status")); boot(); }
          catch (e) { say("boot: " + e.message); console.error(e); }
          return () => { try { mapL && mapL.remove(); mapR && mapR.remove(); } catch (e) {} };
        }
        export default {render};
        """

    return (PairMap,)


@app.cell
def _(
    AOI,
    CT_YEARS,
    FILLS,
    FILL_NAMES,
    FILL_SHORT,
    HEX_ZOOM,
    HL_FLAT,
    HL_LIFT,
    HL_MID,
    HOME,
    LABELS_SLOT,
    MAX_Z,
    MIN_Z,
    NDVI_HEX,
    NDVI_HI,
    NDVI_LO,
    PIC_MODES,
    PIC_YEARS,
    PairMap,
    RASTER_TILE,
    RING_HOVER,
    RING_PICK,
    RING_PICK_W,
    RING_W,
    SCALE0,
    ADMIN_COUNTY_LINE,
    ADMIN_COUNTY_WIDTH,
    ADMIN_LINE,
    ADMIN_PM,
    ADMIN_STATES,
    ADMIN_WIDTH,
    STRIP_MINIMAL,
    VIEW_H,
    WILD_LINE,
    WILD_WIDTH,
    WIN_FROM0,
    WIN_TO0,
    YEAR0,
    json,
    wild_rings,
):
    # ---- the map: built ONCE, empty; never re-runs for a parameter ---------------
    # the wildland rings cross once, here. That button and ADMIN are
    # browser-side after this, so `wild` and `admin` are never reconciled
    # from config the way the other controls are. `water` is: it is a fold.
    _wild_xy, _wild_idx, _wild_n = wild_rings()
    pair = PairMap(wild_xy=_wild_xy.tobytes(), wild_idx=_wild_idx.tobytes(), config=json.dumps({
        "height": VIEW_H, "home": dict(HOME), "labels": True, "labels_slot": LABELS_SLOT, "tile": RASTER_TILE,
        "year": YEAR0, "pic_years": list(PIC_YEARS), "scale": SCALE0, "scale_gen": 0,
        "pic_mode": "tc", "pic_modes": [list(m) for m in PIC_MODES],
        "ndvi_ramp": NDVI_HEX, "ndvi_lo": NDVI_LO, "ndvi_hi": NDVI_HI,
        "min_z": MIN_Z, "max_z": MAX_Z, "bbox": list(AOI),
        "fill": FILLS[0], "fills": [[f, FILL_SHORT[f], FILL_NAMES[f]] for f in FILLS],
        "win_from": WIN_FROM0, "win_to": WIN_TO0, "win_years": list(CT_YEARS),
        "wild": False, "wild_n": _wild_n,
        "wild_line": list(WILD_LINE), "wild_width": WILD_WIDTH,
        "admin": False, "admin_pm": ADMIN_PM, "admin_states": list(ADMIN_STATES),
        "admin_line": list(ADMIN_LINE), "admin_width": ADMIN_WIDTH,
        "admin_county_line": list(ADMIN_COUNTY_LINE), "admin_county_width": ADMIN_COUNTY_WIDTH,
        "water": True,
        "hex_zoom": HEX_ZOOM,
        "hl_lift": HL_LIFT, "hl_mid": HL_MID, "hl_flat": list(HL_FLAT),
        "ring_hover": list(RING_HOVER), "ring_pick": list(RING_PICK),
        "ring_w": RING_W, "ring_pick_w": RING_PICK_W,
        "minimal": STRIP_MINIMAL,
    }))
    HOLD = {
        "frame": None, "sent": None, "box": None, "res": None, "vs": None,
        "busy": False, "pending": None, "pending_force": False, "task": None, "loop": None,
        "year": YEAR0, "scale": SCALE0, "scale_gen": 0, "fill": FILLS[0], "labels": True, "mode": "tc",
        "y0": WIN_FROM0, "y1": WIN_TO0, "water": True,
        "hit": None, "memo": {}, "ct": {}, "h_cam": None, "h_ctl": None, "h_pick": None,
        "runs": 0,
    }
    pair
    return HOLD, pair


@app.cell
def _(
    AOI,
    CELL_KM2,
    CT_YEARS,
    FILLS,
    FILL_NAMES,
    HEX_ZOOM,
    HOLD,
    HOME,
    PIC_MODES,
    PIC_YEARS,
    PIC_YEARS_T,
    SETTLE,
    STRIP_MINIMAL,
    asyncio,
    build_frame,
    clip_box,
    con,
    contains,
    ct_fold,
    grid,
    json,
    math,
    ndvi_series,
    np,
    pad_box,
    pair,
    pic_set_scale,
    pic_stats,
    pic_tile_png,
    pic_warm,
    res_for_view,
    source_series,
    tiles,
    time,
    traceback,
    view_to_bbox,
    wild_at,
    division_at,
):
    # ---- wiring: the camera loop and the controls. Re-runs freely. ---------------
    try:
        HOLD["loop"] = asyncio.get_running_loop()
    except RuntimeError:
        pass
    HOLD["runs"] += 1

    async def _tile_fn(z, x, y, year, mode="tc", coarse=0, dropped=None):
        return await pic_tile_png(z, x, y, year, mode, coarse, dropped)

    pair.tile_fn = _tile_fn

    async def _warm_fn(z, xys, years, mode="tc", coarse=1):
        def _done(yr, n):
            _say((HOLD.get("last_status") or "") + f" · warmed to {yr} ({n:,} tiles, blocks {tiles.block_hits:,} hit / {tiles.block_fetches:,} fetched)")
        await pic_warm(z, xys, years, mode, coarse, _done)

    pair.warm_fn = _warm_fn

    def _say(msg):
        try:
            pair.status = msg
        except Exception:
            pass

    def _cfg(**kw):
        c = json.loads(pair.config or "{}")
        c.update(kw)
        pair.config = json.dumps(c)

    def _vsd(vs):
        if vs is None:
            return dict(HOME)
        if isinstance(vs, str):
            try:
                vs = json.loads(vs)
            except Exception:
                return dict(HOME)
        out = {"longitude": float(vs["longitude"]), "latitude": float(vs["latitude"]), "zoom": float(vs["zoom"])}
        if vs.get("w") and vs.get("h"):
            out["w"], out["h"] = float(vs["w"]), float(vs["h"])
        return out

    def _hexes_off(msg):
        if HOLD["sent"] is not None:
            with pair.hold_sync():
                pair.cells, pair.colors = b"", b""
            HOLD["sent"] = None
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = None, None, None, None
        _cfg(hit=None)
        pair.legend = "[]"
        pair.panel = pair.panel_l = ""
        _say(msg)

    def _paint():
        fr = HOLD["frame"]
        if fr is None:
            return False
        rgba = fr["fill"](HOLD["fill"], HOLD["hit"])
        _cfg(hit=format(HOLD["hit"], "x") if HOLD["hit"] else None)
        with pair.hold_sync():
            if HOLD["sent"] is not fr:
                pair.cells = fr["cellid"].astype("<u8").tobytes()
                HOLD["sent"] = fr
            pair.colors = rgba.tobytes()
        pair.legend = json.dumps(fr["legend"](HOLD["fill"]))
        return True

    async def _serve(vs, force=False):
        vsd = _vsd(vs)
        view = view_to_bbox(vsd)
        if vsd["zoom"] < HEX_ZOOM:
            _hexes_off(f"zoom {vsd['zoom']:.1f} · zoom in past {HEX_ZOOM:g} for the hexagons (CTrees has no pyramid to draw below it)")
            return
        box = clip_box(pad_box(view), AOI)
        if box is None:
            _hexes_off("the view is outside the New England AOI; CTrees is only read inside it")
            return
        inside = HOLD["box"] is not None and contains(HOLD["box"], clip_box(view, AOI) or view)
        if inside and res_for_view(vsd, box) <= HOLD["res"]:
            if not force:
                _say(HOLD.get("last_status", "") + " · held")
                return
            # a window change inside the held box keeps the held box and res,
            # so the biomass window read is a cache hit and only the two
            # uncertainty ends are new
            box, res = HOLD["box"], HOLD["res"]
        else:
            res = res_for_view(vsd, box)
        y0, y1 = HOLD["y0"], HOLD["y1"]
        rbox = tuple(round(v, 3) for v in box)
        water = HOLD["water"]
        key = (y0, y1, res, rbox, water)
        t0 = time.time()
        _say("folding CTrees, all 26 years…" if STRIP_MINIMAL else f"res {res} · folding CTrees 2000..2025…")
        if key in HOLD["memo"]:
            fr, stats = HOLD["memo"][key]
        else:
            bkey = (res, rbox, y0, y1, water)
            if bkey not in HOLD["ct"]:
                HOLD["ct"][bkey] = await ct_fold(box, res, y0, y1, water)
                if len(HOLD["ct"]) > 24:
                    HOLD["ct"].pop(next(iter(HOLD["ct"])))
            tab, s1 = HOLD["ct"][bkey]
            if tab is None or tab.num_rows == 0:
                _say(f"res {res} · {s1}")
                return
            t1 = time.time()
            loop = asyncio.get_running_loop()
            fr = await loop.run_in_executor(None, build_frame, tab, y0, y1)
            stats = f"res {res} · {s1} · frame {time.time() - t1:.1f} s"
            HOLD["memo"][key] = (fr, stats)
            if len(HOLD["memo"]) > 24:
                HOLD["memo"].pop(next(iter(HOLD["memo"])))
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = fr, box, res, None
        t2 = time.time()
        _paint()
        st = pic_stats()
        HOLD["last_status"] = (
            f"{stats} · {fr['score']}"
            + f" · send {time.time() - t2:.2f} s · {time.time() - t0:.1f} s"
            f" · mosaic tiles {st['served']:,} drawn, {st['blank']:,} empty, {st.get('warmed', 0):,} warmed"
        )
        _say(HOLD["last_status"])

    async def refresh(vs, force=False, settle=True):
        """ONE serve at a time; the latest request wins while one is in flight."""
        if HOLD["busy"]:
            HOLD["pending"] = vs
            HOLD["pending_force"] = HOLD["pending_force"] or force
            return
        HOLD["busy"] = True
        try:
            while True:
                if settle:
                    await asyncio.sleep(SETTLE)
                if HOLD["pending"] is not None:
                    vs, HOLD["pending"] = HOLD["pending"], None
                    force, HOLD["pending_force"] = HOLD["pending_force"], False
                    settle = True
                    continue
                await _serve(vs, force)
                vs = HOLD["pending"]
                if vs is None:
                    return
                force, HOLD["pending"], HOLD["pending_force"] = HOLD["pending_force"], None, False
                settle = False
        except Exception as exc:
            tb = traceback.extract_tb(exc.__traceback__)
            where = f" (line {tb[-1].lineno})" if tb else ""
            _say(f"failed: {type(exc).__name__}: {exc}{where}")
            raise
        finally:
            HOLD["busy"], HOLD["pending"], HOLD["pending_force"] = False, None, False

    def _spawn(coro):
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            loop = HOLD.get("loop")
            return asyncio.run_coroutine_threadsafe(coro, loop) if loop else None

    def _request(force=False):
        vs = HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)
        HOLD["task"] = _spawn(refresh(vs, force, settle=False))

    def _on_camera(change):
        vs = change["new"]
        if not vs:
            return
        HOLD["vs"] = vs
        HOLD["task"] = _spawn(refresh(vs))

    if HOLD.get("h_cam") is not None:
        try:
            pair.unobserve(HOLD["h_cam"], names="view")
        except ValueError:
            pass
    pair.observe(_on_camera, names="view")
    HOLD["h_cam"] = _on_camera

    def _f(v, d=1):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"

    def _spark(series, y0, y1, unc=None):
        """A small inline SVG of the 26-year series, the window shaded."""
        W, H, pad = 260, 56, 4
        ys = np.array(series, np.float64)
        ok = np.isfinite(ys)
        if not ok.any():
            return ""
        top = max(float(np.nanmax(ys)), 1.0)
        n = len(ys)
        xs = [pad + i * (W - 2 * pad) / (n - 1) for i in range(n)]
        pts = " ".join(f"{xs[i]:.1f},{H - pad - (ys[i] / top) * (H - 2 * pad):.1f}" for i in range(n) if ok[i])
        i0, i1 = CT_YEARS.index(y0), CT_YEARS.index(y1)
        band = ""
        if unc is not None:
            us = np.array(unc, np.float64)
            hi = " ".join(f"{xs[i]:.1f},{H - pad - min(1.0, (ys[i] + us[i]) / top) * (H - 2 * pad):.1f}" for i in range(n) if ok[i] and np.isfinite(us[i]))
            lo = " ".join(f"{xs[i]:.1f},{H - pad - max(0.0, (ys[i] - us[i]) / top) * (H - 2 * pad):.1f}" for i in reversed(range(n)) if ok[i] and np.isfinite(us[i]))
            band = f"<polygon points='{hi} {lo}' fill='#2a5db0' opacity='.12'/>"
        return (
            f"<svg width='{W}' height='{H}' style='vertical-align:middle;margin-left:.6rem'>"
            f"<rect x='{xs[i0]:.1f}' y='{pad}' width='{xs[i1] - xs[i0]:.1f}' height='{H - 2 * pad}' fill='#e07b1c' opacity='.13'/>"
            f"{band}<polyline points='{pts}' fill='none' stroke='#2a5db0' stroke-width='1.5'/>"
            f"<text x='{xs[0]:.0f}' y='{H - 1}' font-size='8' fill='#888'>{CT_YEARS[0]}</text>"
            f"<text x='{xs[-1] - 18:.0f}' y='{H - 1}' font-size='8' fill='#888'>{CT_YEARS[-1]}</text>"
            f"<text x='{pad}' y='9' font-size='8' fill='#888'>{top:.0f} Mg/ha</text></svg>"
        )

    def _ndvi_spark(series, year, lv, npx):
        """The clicked cell's NDVI change from the first mosaic year, every
        year, on a symmetric axis whose bound is printed.

        The level is not the story: New England forest sits at 0.82 to 0.90,
        so on the fixed NDVI_LO..NDVI_HI scale the series was a flat line at
        the ceiling and the caption carried all of it. The delta spends the
        axis on the only part that moves, and printing the bound keeps two
        clicks honest."""
        # the caption sits in its own row UNDER the plot: forest NDVI pins to
        # the top of the fixed scale, so anything drawn inside the plot box
        # collides with the line
        W, H, pad = 260, 64, 4
        FOOT = 14                       # the label row, outside the plot
        top, bot = pad, H - FOOT
        ys = np.array(series, np.float64)
        ok = np.isfinite(ys)
        if not ok.any():
            return "<span style='opacity:.7'>No valid mosaic pixel here.</span>"
        n = len(ys)
        xs = [pad + i * (W - 2 * pad) / (n - 1) for i in range(n)]
        first, last = ys[ok][0], ys[ok][-1]
        ds = ys - first                       # the plotted quantity: change
        # symmetric bound in 0.05 steps, floored at 0.05 so a cell that truly
        # does nothing draws as a flat line and not as noise filling the box
        mx = float(np.nanmax(np.abs(ds[ok])))
        B = max(0.05, float(np.ceil(mx * 20.0) / 20.0))
        mid = (top + bot) / 2.0

        def yy(v):
            return mid - min(1.0, max(-1.0, v / B)) * (bot - top) / 2.0

        pts = " ".join(f"{xs[i]:.1f},{yy(ds[i]):.1f}" for i in range(n) if ok[i])
        # The marker moves in the browser, not here. styleYear() reads these
        # two arrays off the svg and slides #ndmk / #nddot, so holding [ or ]
        # scrubs the marker across the chart with no kernel round trip, the
        # same way the wildlands toggle never reaches the kernel.
        mk_x = json.dumps([round(float(x), 1) for x in xs])
        mk_y = json.dumps([round(float(yy(ds[i])), 1) if ok[i] else None for i in range(n)])
        i = PIC_YEARS_T.index(year) if year in PIC_YEARS_T else 0
        hid = "" if ok[i] else " visibility='hidden'"
        cy = f"{yy(ds[i]):.1f}" if ok[i] else "-9"
        mark = (f"<line id='ndmk' x1='{xs[i]:.1f}' y1='{top}' x2='{xs[i]:.1f}' y2='{bot}' stroke='#2a5db0' stroke-width='1' opacity='.45'/>"
                f"<circle id='nddot' cx='{xs[i]:.1f}' cy='{cy}' r='2.6' fill='#2a5db0'{hid}/>")
        d = last - first
        way = "up" if d > 0 else "down"
        line = (f"Mosaic NDVI: <b>{first:.2f}</b> in {PIC_YEARS_T[0]}, <b>{last:.2f}</b> in "
                f"{PIC_YEARS_T[-1]}: {way} <b>{abs(d):.2f}</b>.")
        detail = "" if STRIP_MINIMAL else (
            f"<div style='font-size:12px;color:#777'>level {lv}, "
            f"{grid.RES * 2 ** lv * 111_000:.0f} m pixels, {npx} sampled</div>")
        return (
            f"<div style='font-size:14px;line-height:1.5'>{line}"
            f"<svg width='{W}' height='{H}' data-mkx='{mk_x}' data-mky='{mk_y}' style='vertical-align:middle;margin-left:.6rem'>"
            f"<line x1='{pad}' y1='{mid:.1f}' x2='{W - pad}' y2='{mid:.1f}' stroke='#ccc' stroke-width='1'/>"
            f"{mark}<polyline points='{pts}' fill='none' stroke='#2a5db0' stroke-width='1.5'/>"
            f"<text x='{pad}' y='{top + 7}' font-size='8' fill='#888'>+{B:.2f}</text>"
            f"<text x='{pad}' y='{bot - 1:.0f}' font-size='8' fill='#888'>-{B:.2f}</text>"
            f"<text x='{xs[0]:.0f}' y='{H - 3}' font-size='8' fill='#888'>{PIC_YEARS_T[0]}</text>"
            f"<text x='{W / 2:.0f}' y='{H - 3}' font-size='8' fill='#888' text-anchor='middle'>NDVI change from {PIC_YEARS_T[0]}</text>"
            f"<text x='{xs[-1] - 18:.0f}' y='{H - 3}' font-size='8' fill='#888'>{PIC_YEARS_T[-1]}</text>"
            f"</svg></div>{detail}")

    def _source_note(counts, npx):
        """One line beside the NDVI chart: where the slider year's pixels came
        from (the browser moves it with the slider, like the NDVI marker), and
        the share borrowed from a neighbouring year over the ladder years.
        The panel stays the height of the chart."""
        if counts is None or not npx:
            return ""
        n = counts.shape[0]
        own, before, after, l7, nod, unm, nc = (counts[:, k].astype(float) for k in range(7))
        seen = np.maximum(1.0, own + before + after + l7 + nod + unm)
        hollow = nc >= 0.5 * npx
        caps = []
        for i in range(n):
            y = PIC_YEARS_T[i]
            if hollow[i]:
                caps.append(f"{y}: own-year pixels, not in the ladder")
                continue
            b, a, o, l, z = (100 * v[i] / seen[i] for v in (before, after, own, l7, nod + unm))
            parts = []
            if o + l >= 0.5:
                parts.append(f"{o + l:.0f}% own year" + (f" ({l:.0f}% Landsat 7)" if l >= 0.5 else ""))
            if b >= 0.5:
                parts.append(f"{b:.0f}% from {y - 1}")
            if a >= 0.5:
                parts.append(f"{a:.0f}% from {y + 1}")
            if z >= 0.5:
                parts.append(f"{z:.0f}% no data")
            caps.append(f"{y}: " + ", ".join(parts))
        computed = ~hollow
        tot = float((seen * computed).sum()) or 1.0
        borrowed = 100 * float(((before + after) * computed).sum()) / tot
        i0 = PIC_YEARS_T.index(HOLD["year"]) if HOLD["year"] in PIC_YEARS_T else 0
        data = json.dumps(caps).replace("'", "&#39;")
        return (f"<span style='display:inline-block;vertical-align:middle;margin-left:.7rem;font-size:12px;line-height:1.35;color:#444;max-width:17rem'>"
                f"<span id='srccap' data-caps='{data}'>{caps[i0]}</span><br>"
                f"<span style='color:#777'>{borrowed:.0f}% of pixels borrowed from a neighbouring year over the {int(computed.sum())} ladder years</span></span>")

    def _on_pick(change):
        fr = HOLD["frame"]
        try:
            p = json.loads(change["new"] or "{}")
        except Exception:
            return
        if fr is None:
            return
        try:
            cellh = p.get("cell")
            if not cellh:
                HOLD["hit"] = None
                pair.panel = pair.panel_l = ""
                _paint()
                return
            cell = int(cellh, 16)
            con.register("cur_cells", fr["cells"])
            r = con.execute(
                "SELECT stock0, stock, change, z, lossyear, lost, recovered, ymin, unc0, unc1, ushare "
                "FROM cur_cells WHERE cell = ?", [cell]
            ).fetchone()
            lat, lon = p.get("lat"), p.get("lon")
            where = f" at {lat:.4f}, {lon:.4f}" if lat is not None and lon is not None else ""
            if r is None:
                HOLD["hit"] = None
                pair.panel = f"<span style='opacity:.7'>{cellh}{where}: not in the current frame</span>"
                pair.panel_l = ""
            else:
                HOLD["hit"] = cell if HOLD["hit"] != cell else None
                s0, s1, ch, z, ly, lost, rec, ymin, u0, u1, ush = r
                y0, y1 = fr["y0"], fr["y1"]
                ci = int(np.searchsorted(fr["cellid"], np.uint64(cell)))
                series = fr["A"][:, ci] if ci < len(fr["cellid"]) else []
                useries = None  # the uncertainty is read for the window's ends only
                if s0 is None or s1 is None or np.isnan(s0) or np.isnan(s1):
                    l1 = "CTrees has no valid biomass here at one end of the window."
                else:
                    how = "down" if ch < 0 else "up"
                    pct = f" ({100 * abs(ch) / s0:.0f}%)" if s0 > 0 else ""
                    ha = CELL_KM2.get(HOLD["res"], 0) * 100.0
                    l1 = (f"CTrees: <b>{s0:.0f} Mg/ha</b> in {y0}, <b>{s1:.0f}</b> in {y1}: "
                          f"{how} <b>{abs(ch):.0f}</b>{pct} over <b>{ha:,.1f} ha</b> "
                          f"(about {abs(ch) * ha:,.0f} Mg across the cell).")
                if ly is not None and ly >= 0:
                    l2 = f"The biomass fell past the threshold in <b>{ly}</b> (the deepest drop inside the window {_f(lost, 0)} Mg/ha)"
                    l2 += f"; {_f(rec, 0)} Mg/ha came back since its low in {ymin}." if rec is not None and rec > 1 else "; nothing came back."
                else:
                    l2 = f"No fall past the threshold {y0} to {y1}" + (f"; the low was {ymin}, {_f(rec, 0)} Mg/ha below {y1}." if rec is not None and rec > 1 else ".")
                if u1 is not None and not np.isnan(u1):
                    zt = "well past" if z >= 2 else ("past" if z >= 1 else "within")
                    l3 = f"Uncertainty ±{u1:.0f} Mg/ha at {y1} ({100 * min(ush, 9.99):.0f}% of the stock); the change is {zt} it ({_f(z, 1)}×)."
                else:
                    l3 = "No uncertainty here."
                # where: the town, county and state. State and county came
                # with the click from the admin tiles in the browser; the
                # town is a query against Source Cooperative, so it lands
                # after, like the NDVI series, if the cell is still picked.
                adm = p.get("admin") or {}

                def _place(town, county, st, pending):
                    bits = []
                    if town:
                        bits.append(f"<b>{town}</b>")
                    elif pending:
                        bits.append("<span style='opacity:.5'>town…</span>")
                    if county:
                        bits.append(county)
                    if st:
                        bits.append(st)
                    return ", ".join(bits)

                l0 = _place(None, adm.get("county"), adm.get("state_code"), True) if lon is not None else ""
                wl = wild_at(lon, lat) if lon is not None and lat is not None else None
                if wl:
                    yr = f"since {wl['year']}" if wl["year"] > 0 else "origin year unrecorded"
                    l4 = (f"<br>Wildland: <b>{wl['name']}</b> ({wl['owner']}, {wl['state']}, "
                          f"{wl['acres']:,.0f} ac, {yr}).")
                else:
                    l4 = ""
                detail = f"{CELL_KM2.get(HOLD['res'], 0):.3f} km²{where}"
                pair.panel = (
                    f"<div style='font-size:14px;line-height:1.5'>{l0}{'<br>' if l0 else ''}{l1}<br>{l2}<br>{l3}{l4}{_spark(series, y0, y1, useries)}</div>"
                    + ("" if STRIP_MINIMAL else f"<div style='font-size:12px;color:#777'>{detail}</div>")
                )
                # The ring and the CTrees story go out now. The mosaic NDVI
                # series is 26 years x 2 bands from the pyramid, 1 to 2.5 s
                # over http (0.02 s when the pyramid was on disk), so it is
                # read off the loop and the left panel filled when it lands,
                # if the same cell is still picked. Holding the ring back
                # behind that read made a click look ignored, and a second
                # click on the same cell then unpicked it.
                pair.panel_l = "<span style='opacity:.6'>reading the mosaic NDVI series…</span>"
                _paint()
                side_m = math.sqrt(max(CELL_KM2.get(HOLD["res"], 0.1053), 1e-9)) * 1000.0

                async def _ndvi_later(cell=cell, lon=lon, lat=lat, side_m=side_m):
                    # the NDVI chart the moment its series lands; the source
                    # note joins it beside the chart when its own read (level
                    # 0, 26 years) is done, without holding the chart back
                    t_nd = asyncio.to_thread(ndvi_series, lon, lat, side_m)
                    t_sc = asyncio.to_thread(source_series, lon, lat, side_m)
                    try:
                        nd, ndlv, ndpx = await t_nd
                    except Exception as e:
                        nd, ndlv, ndpx = None, 0, 0
                        _say(f"NDVI series: {e}")
                    if HOLD["hit"] != cell:
                        return
                    spark = _ndvi_spark(nd, HOLD["year"], ndlv, ndpx) if nd is not None else ""
                    pair.panel_l = spark
                    try:
                        sc, spx = await t_sc
                    except Exception as e:
                        sc, spx = None, 0
                        _say(f"source series: {e}")
                    if HOLD["hit"] != cell or not spark:
                        return
                    pair.panel_l = spark.replace("</svg></div>", "</svg>" + _source_note(sc, spx) + "</div>", 1)

                _spawn(_ndvi_later())

                async def _place_later(cell=cell, lon=lon, lat=lat, l0=l0, adm=adm):
                    if not l0:
                        return
                    try:
                        d = await asyncio.to_thread(division_at, lon, lat)
                    except Exception as e:
                        d = {}
                        _say(f"town (fused/overture): {e}")
                    if HOLD["hit"] != cell:
                        return
                    l0n = _place(d.get("town"), adm.get("county") or d.get("county"),
                                 adm.get("state_code") or d.get("state_code"), False)
                    pair.panel = pair.panel.replace(l0, l0n, 1)

                _spawn(_place_later())
                return
        except Exception as e:
            pair.panel = f"<span style='opacity:.7'>click: {e}</span>"
            pair.panel_l = ""
        _paint()

    if HOLD.get("h_pick") is not None:
        try:
            pair.unobserve(HOLD["h_pick"], names="pick")
        except ValueError:
            pass
    pair.observe(_on_pick, names="pick")
    HOLD["h_pick"] = _on_pick

    def _on_ctl_body(change):
        try:
            c = json.loads(change["new"] or "{}")
        except Exception:
            return
        act = c.get("act")
        if act == "year":
            y = int(c.get("year", HOLD["year"]))
            if y in PIC_YEARS and y != HOLD["year"]:
                HOLD["year"] = y
                _cfg(year=y)
                _say((HOLD.get("last_status") or "") + f" · mosaic {y}")
            return
        if act == "scale":
            try:
                v = float(min(3.0, max(0.2, float(c.get("scale", HOLD["scale"])))))
            except (TypeError, ValueError):
                return
            if pic_set_scale(v):
                HOLD["scale"] = v
                HOLD["scale_gen"] += 1
                _cfg(scale=v, scale_gen=HOLD["scale_gen"])
                _say((HOLD.get("last_status") or "") + f" · picture scale {v:.1f}× · tiles re-served")
            return
        if act == "win":
            a, b = int(c.get("y0", HOLD["y0"])), int(c.get("y1", HOLD["y1"]))
            if a in CT_YEARS and b in CT_YEARS and a < b and (a, b) != (HOLD["y0"], HOLD["y1"]):
                HOLD["y0"], HOLD["y1"] = a, b
                _cfg(win_from=a, win_to=b)
                _request(force=True)
            return
        if act == "fill":
            f = c.get("fill")
            if f in FILLS and f != HOLD["fill"]:
                HOLD["fill"] = f
                _cfg(fill=f)
                if _paint():
                    _say((HOLD.get("last_status") or "") + f" · {FILL_NAMES[f]}")
            return
        if act == "mode":
            m = c.get("mode", "tc")
            if m in dict(PIC_MODES) and m != HOLD["mode"]:
                HOLD["mode"] = m
                _cfg(pic_mode=m)
                _say((HOLD.get("last_status") or "") + f" · picture: {dict(PIC_MODES)[m]}")
            return
        if act == "labels":
            HOLD["labels"] = bool(c.get("labels", True))
            _cfg(labels=HOLD["labels"])
            return
        if act == "water":
            w = bool(c.get("water", True))
            if w != HOLD["water"]:
                HOLD["water"] = w
                _cfg(water=w)
                _request(force=True)
            return

    def _on_ctl(change):
        try:
            _on_ctl_body(change)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            where = f" (line {tb[-1].lineno})" if tb else ""
            _say(f"control failed: {type(e).__name__}: {e}{where}")

    if HOLD.get("h_ctl") is not None:
        try:
            pair.unobserve(HOLD["h_ctl"], names="ctl")
        except ValueError:
            pass
    pair.observe(_on_ctl, names="ctl")
    HOLD["h_ctl"] = _on_ctl

    if HOLD["frame"] is None and not HOLD["busy"]:
        _request()
    else:
        _paint()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Under the map

    DuckDB over the CURRENT view's cells (press the button after the map
    settles): `npx` (sampled CTrees pixels), `stock0` / `stock` (Mg/ha at the
    window's two ends), `change`, `z` (the change over the uncertainty of
    its two ends), `lossyear` (-1 none), `lost` (the deepest drop from the
    running maximum), `recovered` (since the window's minimum, in `ymin`),
    `unc0` / `unc1`, `ushare` (uncertainty over stock at the to-end).
    """)
    return


@app.cell
def _(mo):
    tables_btn = mo.ui.run_button(label="tables for the current view")
    tables_btn
    return (tables_btn,)


@app.cell
def _(HOLD, con, mo, tables_btn):
    mo.stop(not tables_btn.value or HOLD["frame"] is None, mo.md("*no view folded yet*"))
    con.register("view_cells", HOLD["frame"]["cells"])
    by_year = mo.sql(
        """
        SELECT lossyear, count(*) AS cells, round(avg(lost), 1) AS mean_drop, round(avg(stock0), 1) AS mean_stock_from,
               round(avg(stock), 1) AS mean_stock_to
        FROM view_cells GROUP BY lossyear ORDER BY lossyear
        """,
        engine=con,
    )
    return


if __name__ == "__main__":
    app.run()
