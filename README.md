# pdnart: make art for paint.net with code or with Claude

paint.net has no scripting API. `pdnart` gives you one another way. You describe a picture as a
**layered scene** (shapes, brush strokes, gradients, scattered stars, text, blur). pdnart
renders the scene and rebuilds it inside paint.net as a real multi-layer document that you can
keep editing by hand.

![example](examples/moonlit_mountains.png)

*`examples/moonlit_mountains.json`: seven layers (sky, stars, moon glow, moon, two mountain
ranges, lake).*

## How it talks to paint.net

pdnart does not click around the canvas pixel by pixel, because that approach is slow and
breaks easily. It uses the two things paint.net reliably supports:

1. **The clipboard.** Each layer is rendered at full canvas size and placed on the clipboard as a
   PNG, which keeps transparency.
2. **Keyboard shortcuts**, sent with `pywinauto`:
   - `Ctrl+Alt+V` (Paste into New Image) for the background
   - `Ctrl+Shift+V` (Paste into New Layer) for each scene layer
   - `F4` (Layer Properties) to name the layer
   - `Ctrl+Shift+S` to optionally save a `.pdn`

Limitations: layer opacity is baked into the pasted pixels, hidden layers are skipped, and blend
modes aren't used. Don't touch the mouse or keyboard while pdnart sends a scene.

## Install

```bash
pip install -e ".[windows,mcp]"   # on the Windows machine that has paint.net
pip install -e .                  # render-only, any OS
```

If paint.net isn't in `C:\Program Files\paint.net\`, set the `PDN_EXE` environment variable to
your `paintdotnet.exe`.

## Use it from the command line

```bash
pdnart render examples/moonlit_mountains.json -o art.png --layers layers/   # any OS
pdnart send   examples/moonlit_mountains.json --save C:\art\moon.pdn         # Windows + paint.net
```

If paint.net is slow on your machine and steps get skipped, increase `--delay` (seconds between
UI steps, default 0.6).

## Let Claude paint (MCP)

Add the server to Claude Desktop (`claude_desktop_config.json`):

```json
{ "mcpServers": { "pdnart": { "command": "pdnart", "args": ["mcp"] } } }
```

or to Claude Code: `claude mcp add pdnart -- pdnart mcp`.

Then ask something like *"Paint a foggy harbour at dawn and open it in paint.net."* Claude gets
these tools:

| tool | what it does |
|---|---|
| `op_reference` | lists the drawing ops |
| `new_canvas`, `add_layer`, `set_layer` | set up the canvas and layers |
| `draw`, `undo` | add or remove ops on a layer |
| `preview` | returns the rendered image so Claude can look at its work and refine it |
| `get_scene`, `load_scene`, `save_files` | JSON, PNG and JPG in and out |
| `send_to_paintnet` | builds the layered document in paint.net, optionally saving a `.pdn` |

## Scene format

```json
{
  "width": 800, "height": 500, "background": "#0b1033",
  "layers": [
    {"name": "Moon", "opacity": 1.0, "visible": true, "ops": [
      {"op": "ellipse", "cx": 610, "cy": 115, "rx": 38, "ry": 38, "fill": "#fff6d8"}
    ]}
  ]
}
```

Layers are listed back to front. Colors can be `#rrggbb`, `#rrggbbaa`, CSS names, or
`[r, g, b, a]`.

| op | required | optional |
|---|---|---|
| `rect` | x, y, w, h | fill, stroke, width, radius |
| `ellipse` | cx, cy, rx, ry | fill, stroke, width |
| `polygon` | points | fill, stroke, width |
| `line` | points | color, width |
| `stroke` (smooth brush) | points | color, width, smooth |
| `gradient` | stops `[[0..1, color], …]` | kind (`linear`: x0,y0,x1,y1 / `radial`: cx,cy,r) |
| `scatter` | count | region `[x,y,w,h]`, radius `[min,max]`, colors, seed, shape (`circle`/`star`) |
| `text` | x, y, text | size, color, font (.ttf path), anchor |
| `blur` | radius | – (blurs what's already on the layer) |
| `fill` | color | – |

## Development

```bash
pip install -e ".[dev,mcp]" && pytest
```

The renderer, the scene model and the MCP server are tested on any OS. The paint.net driver
(`pdnart/paintnet.py`) only runs on Windows with paint.net installed.
