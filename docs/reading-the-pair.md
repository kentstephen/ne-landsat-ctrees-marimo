# Reading the pair

Two maps in one widget, one camera. Pan or zoom either and both move.
Hover a hexagon on either side and its outline is drawn on both.

## The left pane: the mosaic

The picture is the annual leaf-on composite from the pyramid, never
covered by a fill. Tiles are rendered in the kernel: at each zoom the
pyramid level whose pixel is the finest one not smaller than the screen
pixel, so a zoomed-out view is a few chunks and a zoomed-in one reads
level 0 at 30 m.

- **Year** (the slider, `[` and `]`, or the arrow keys): 2000 to 2025.
  Step quickly through the years and the picture is drawn one level up
  while you move, then sharpened for the year you stop on; the kernel
  fetches the years ahead of you in the direction you are going.
  `b` blinks between the first and last year.
- **Show** (`n`): true colour, or NDVI on a fixed -0.1 to 0.9 scale.
  **Scale** is a gain on true colour under a gamma of 2.
- **Find**: a place name; Enter or a click flies both maps there.
- **Wildlands** (`w`): the 426 Wildlands of New England (2022), conserved
  land left to natural process, drawn as orange boundaries on both panes.
- **Admin** (`s`): state and county boundaries from Overture Maps
  divisions, read live from Source Cooperative as PMTiles, charcoal on
  both panes. The basemap's own admin lines are hidden.
- **Water mask** (`m`, on by default): lakes, ponds, reservoirs, bays and
  wide rivers of 1 ha and up leave the CTrees fold at the pixel level, so
  no hexagon is drawn over open water and a shore hexagon averages its
  land pixels only. Off, water reads as zero biomass.
- `L` toggles the basemap labels, `F` full screen.

Both panes are clipped to the six states. The pyramid itself is not
masked; its tiles are whole squares that keep New York, Quebec, New
Brunswick and the ocean.

## The right pane: CTrees folded to H3

One opaque fill per hexagon at H3 resolution 9 and coarser, from CTrees
global aboveground biomass: 100 m, annual 2000 to 2025, Mg/ha, with a
residual standard error per pixel and year. All 26 years are read for the
box on screen once; the window picks two ends and every window change is
a frame, never a fetch. The hexagons fold from zoom 9 and stop at
resolution 9, about one CTrees pixel, while the picture keeps zooming.
Below zoom 9 the right pane is empty.

- **Fill** (`1` and `2`): **change**, the biomass at the window's to-end
  minus the from-end, blue where it rose and orange where it fell, faint
  where the change is smaller than the uncertainty of its two ends;
  **stock**, the biomass at the to-end on greens, faint where the
  uncertainty exceeds it.
- **Window** (the slider, `-` `=` for the from-end and `_` `+` for the
  to-end): whole years 2000 to 2025.

The uncertainty is opacity throughout: a change smaller than the
uncertainty of its two ends is drawn faint, and so is a stock the
uncertainty exceeds.

## A click

Click a hexagon on either side and it is outlined in gold on both.

Under the left pane: the cell's mosaic NDVI for every year, plotted as
the change from 2000 on a symmetric axis whose bound is printed, with the
slider's year marked; the marker follows the slider. Beside the chart,
where the slider year's pixels came from: the composite lends looks from
the neighbouring years where a year has too few clear looks of its own,
and the pyramid's `source` plane records it per pixel. The note names the
share from the year before and the year after for the slider's year, and
the share borrowed over all the years the rule was applied to.

Under the right pane, first the place: the town, county and state the
click fell in, from Overture Maps divisions on Source Cooperative. The
state and county are read in the browser from the admin tiles already on
screen; the town is one DuckDB point query against the divisions
GeoParquet in fused/overture, so it lands a second or two after the rest.
Then the CTrees stock at both window ends, the change in Mg/ha and in Mg
across the cell's hectares, the year the biomass first fell past the
threshold and what came back since, the uncertainty at the to-end and how
the change compares with it, the 26-year biomass series, and the wildland
the click fell in, if any.

Both series share the same 26 years, so the two charts line up.

## Under the map

A button runs DuckDB over the cells of the current view and prints the
table behind the fills: sampled pixels, stock at both ends, change, the
change over its uncertainty, loss year, deepest drop, recovery, and the
uncertainty share.
