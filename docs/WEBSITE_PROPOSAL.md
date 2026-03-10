# Open-CITE Website Proposal

## Hosting

Static site hosted on **GitHub Pages** from the `/website` directory in the repo root, deployed via GitHub Actions. Built with plain HTML/CSS/JS — no static site generator required. Custom domain: `open-cite.org`.

---

## Brand Identity

### Logo
The Open-CITE bear mascot (polar bear with magnifying glass and goggles) is the primary visual identity. Used in the hero section, favicon, and footer.

### Color Palette (derived from the application's Open-CITE theme)

| Role             | Color     | Usage                                    |
|------------------|-----------|------------------------------------------|
| Primary Blue     | `#2563eb` | Buttons, links, accents                  |
| Dark Blue        | `#1e3a5f` | Gradient backgrounds, hero section       |
| Secondary Blue   | `#1e40af` | Hover states, section headings           |
| White            | `#ffffff` | Cards, text on dark backgrounds          |
| Dark Text        | `#111827` | Body copy on light backgrounds           |
| Muted Text       | `#6b7280` | Captions, secondary text                 |
| Accent Green     | `#10b981` | Status indicators, "powered by" callouts |
| LangGuard Green  | `#00e676` | LangGuard branding elements              |

### Typography
System font stack matching the application: `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`

---

## Site Map

```
/                   Home (single-page, scrollable sections)
```

This is a **single-page website** with smooth-scroll navigation anchors:

1. Hero
2. What is Open-CITE?
3. Key Features
4. Supported Platforms
5. Screenshots
6. Getting Started
7. Plugin Ecosystem
8. Links & Resources
9. Footer

---

## Section-by-Section Design

### 1. Navigation Bar (sticky)

- Open-CITE bear logo (small) + "Open-CITE" text on the left
- Anchor links: Features | Platforms | Screenshots | Get Started
- Right side: GitHub icon link + "Star on GitHub" button
- Collapses to hamburger on mobile

### 2. Hero Section

**Background:** Blue gradient (`#1e3a5f` to `#2563eb`) matching the app's header.

**Content:**
- Large bear mascot logo (centered or left-aligned)
- Headline: **"Open-CITE"** in large white text
- Subline: **"Open Catalog of Intelligent Tools in the Enterprise"**
- One-sentence pitch: Discover and catalog every AI tool, model, and agent across your enterprise — from a single pane of glass.
- Two CTA buttons:
  - **"Get Started"** (white button, scrolls to Getting Started section)
  - **"View on GitHub"** (outline button, links to GitHub repo)
- "powered by LangGuard.AI" badge with LangGuard logo beneath the CTAs

### 3. What is Open-CITE?

**Layout:** Centered text block with max-width, on white background.

**Content:**
- Brief explanation (3-4 sentences) of what Open-CITE does
- Emphasize: Python library + service + GUI, plugin-based, multi-cloud
- Highlight that it runs as a library, service, GUI, or headless API in Docker/Kubernetes

### 4. Key Features

**Layout:** 2x3 grid of feature cards on a light gray background.

**Cards** (icon + title + short description):

| Icon | Title | Description |
|------|-------|-------------|
| Magnifying glass | Multi-Platform Discovery | Automatic discovery across Databricks, AWS, Google Cloud, Azure, and more |
| Signal bars | OpenTelemetry Native | Built-in OTLP receiver for real-time trace collection and analysis |
| Puzzle piece | Plugin Architecture | Add support for any platform with the extensible plugin system |
| Network nodes | Lineage Graphs | Visualize relationships between tools, models, and agents |
| Download | Unified Schema Export | Standardized JSON export for downstream processing and governance |
| Server | Flexible Deployment | Run as a library, GUI, headless API, Docker container, or in Kubernetes |

### 5. Supported Platforms

**Layout:** Horizontal row of platform logos/badges on white background.

**Platforms** (each with a logo/icon and name):
- Databricks
- AWS Bedrock
- AWS SageMaker
- Google Cloud / Vertex AI
- Azure AI Foundry
- Microsoft Fabric
- OpenTelemetry
- MCP (Model Context Protocol)
- Splunk
- Zscaler

Each logo links down to the Plugin Ecosystem section for more detail.

### 6. Screenshots

**Layout:** Tabbed or carousel display on a blue-tinted background. Each screenshot shown in a browser-frame mockup for polish.

**Screenshots to include (captured from the running application):**

1. **Dashboard Overview** — Hero header with bear logo, stats counters (Tools, Models, Agents, MCP Servers), configured plugins sidebar, and lineage graph
2. **Asset Discovery** — The "All Assets" tab showing the grid of discovered tools with type badges, discovery source, and mapping controls
3. **Models View** — The Models tab showing discovered models with provider and usage counts
4. **Agents View** — The Agents tab showing discovered agents with their tools and models listed
5. **Add Plugin Dialog** — The modal showing all available plugin types (AWS Bedrock, SageMaker, Azure AI Foundry, Databricks, Google Cloud, OpenTelemetry)
6. **Lineage Graph** — Close-up of the network visualization showing relationships between assets

Each screenshot gets a short caption explaining what the user is seeing.

### 7. Getting Started

**Layout:** Stepped instructions on white background, with code blocks.

**Steps:**

```
Step 1: Install
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .

Step 2: Launch the GUI
    python -m open_cite.gui.app
    # Open http://localhost:5000

Step 3: Add a Plugin
    Click "+ Add Plugin" and select your platform

Step 4: Discover
    Assets appear automatically as Open-CITE scans your environment
```

Below the steps: link to full documentation on GitHub.

### 8. Plugin Ecosystem

**Layout:** Accordion or expandable cards on light gray background.

For each plugin, show:
- Plugin name and icon
- 2-3 bullet points of what it discovers
- Link to the plugin's documentation on GitHub

Plugins listed:
- **Databricks** — Unity Catalog, MLflow traces, AI Gateway, Genie
- **AWS Bedrock** — Foundation models, custom models, invocations
- **AWS SageMaker** — Endpoints, models, training jobs
- **Google Cloud** — Vertex AI models, endpoints, generative AI models, MCP servers
- **Azure AI Foundry** — Resources, deployments, projects, agents, tools, traces
- **Microsoft Fabric** — Analytics platform integration
- **OpenTelemetry** — OTLP trace receiver, tool/model discovery from traces
- **MCP** — Server discovery, tool/resource cataloging

### 9. Links & Resources

**Layout:** Two prominent cards side-by-side on white background.

**Card 1 — GitHub**
- GitHub logo
- "Open Source on GitHub"
- "Browse the code, file issues, and contribute"
- Button: "View Repository" -> GitHub repo URL

**Card 2 — LangGuard**
- LangGuard logo
- "Enterprise AI Governance"
- "Want to govern and monitor your AI assets? Try LangGuard's AI Control Plane."
- Button: "Visit LangGuard.AI" -> https://langguard.ai?utm=open-cite-website

### 10. Footer

**Layout:** Dark blue background (`#1e3a5f`), white text.

**Content:**
- Open-CITE bear logo (small) + "Open-CITE" text
- "Open Catalog of Intelligent Tools in the Enterprise"
- "Powered by LangGuard.AI"
- Links: GitHub | Documentation | LangGuard.AI
- Copyright line

---

## Technical Implementation

### Technology Stack
- **Plain HTML/CSS/JS** — no build step, no framework dependency
- **CSS custom properties** for the color palette (easy theme consistency)
- **Responsive design** — mobile-first with breakpoints at 768px and 1024px
- **Intersection Observer** for scroll-triggered fade-in animations
- **GitHub Pages** deployment from the repository

### File Structure
```
website/
├── CNAME               # Custom domain: open-cite.org
├── index.html          # Single-page site (all sections)
├── css/
│   └── style.css       # All styles + responsive
├── js/
│   └── main.js         # Scroll animations, mobile nav, tabs, accordion
└── images/
    ├── open-cite-bear.png
    ├── langguard-logo.png
    ├── favicon.png
    └── screenshots/
        ├── dashboard.png
        ├── assets.png
        ├── models.png
        ├── agents.png
        ├── add-plugin.png
        └── lineage.png
```

Platform SVG logos are inline in `index.html` (geometric icons with brand colors, not trademarked logos).

### GitHub Pages Setup
- Enable GitHub Pages in repo Settings -> Pages -> Source: "GitHub Actions"
- GitHub Actions workflow at `.github/workflows/deploy-website.yml` deploys only the `/website` directory
- CNAME file configures custom domain `open-cite.org`
- DNS: A records pointing to GitHub Pages IPs (`185.199.108-111.153`), CNAME `www` -> `LangGuard-AI.github.io`

### Performance Targets
- Single HTML file, one CSS file, one JS file
- All images optimized (WebP with PNG fallback for the bear logo)
- No external dependencies beyond Google Fonts (optional)
- Target: < 500KB total page weight, Lighthouse score > 95

---

## Content That Needs to Be Provided

1. **GitHub repository URL** — to link the "View on GitHub" button and all GitHub references
2. **LangGuard.AI URL** — confirmed as `https://langguard.ai` (with UTM parameter `?utm=open-cite-website`)
3. **Platform logos** — SVG logos for Databricks, AWS, Google Cloud, Azure, OpenTelemetry (or we use simple text badges)
4. **Custom domain** (optional) — if you want something like `opencite.dev` instead of `username.github.io/open-cite`

---

## Summary

A clean, single-page promotional website that:
- Leads with the bear mascot and blue gradient brand identity
- Clearly explains what Open-CITE does in seconds
- Shows real application screenshots to build credibility
- Makes getting started dead simple with copy-paste instructions
- Highlights the breadth of supported platforms
- Drives traffic to both GitHub (for contributors/users) and LangGuard (for enterprise customers)
- Requires zero build tooling — just static files served by GitHub Pages
