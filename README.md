# HacKSU Presentation Template

A slide deck template to use for HacKSU workshops.

```
.github/workflows/
  pages.yml               builds and deploys to GitHub Pages
static/
  app.js                  slide navigation and particle background
  style.css               HacKSU deck design
  HacKSU.png              logo
templates/
  slides/                 slide files
  base.html               page shell, nav bar, logo
  index.html              slide order
  notes.html              speaker notes and run of show
app.py                    base app, two routes: deck, notes
build.py                  renders the deck to static HTML in dist/
README.md                 documentation
requirements.txt          dependencies
```

A Flask app on port 5174.

Install dependencies: 

`pip install -r requirements.txt`

To run the app:

`python app.py` or `flask app`

`/` is the deck, navigated with arrows, space, or the dots.

`/notes` is the speaker notes with a minute-by-minute run of show.

To change the order of slides, navigate to `templates/index.html`, and reorder using `{% include %}`

## GitHub Pages

Every push to the default branch builds the deck and publishes it to GitHub Pages. After forking, do this once:

1. **Actions** tab: enable workflows (GitHub disables them on forks).
2. **Settings → Pages**: set Source to **GitHub Actions**.
3. **Actions** tab: run **Deploy to GitHub Pages** manually, or push a commit.

The deck is published at `https://<username>.github.io/<repo-name>/` and the notes at `.../notes.html`.

To build locally: `python build.py`, then open `dist/index.html`.

Design and plumbing lifted from the polyglot programming deck.

# HacKSU Computer Vision Lesson
Learn all about how computer vision works, how computer process images, and the different methods of detection that computers have used through the ages. We will cover topics on image preprocessing, classical computer vision solutions such as edge detection, and more modern solutions using Convolutional Neural Networks.

## Package Installation
The lesson uses uv:

bash
```
uv init
uv add opencv-python numpy ultralytics
```

Or with plain pip:

bash
```
pip install opencv-python numpy ultralytics
```

The first run of the YOLO script downloads the yolo11n.pt model weights automatically, this can take some time as the weights a little large.



