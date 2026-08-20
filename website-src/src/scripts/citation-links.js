// Turn "[8]", "[1, 2]", "[56, 80]", "[81–83]" into links to #ref-N
function linkifyCitations(root) {
  const excludeSel =
    ".references, .search-block, a, code, .mono, .url, script, style";
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const text = node.nodeValue;
      if (!text || text.indexOf("[") === -1) return NodeFilter.FILTER_REJECT;
      let el = node.parentElement;
      while (el && el !== root) {
        if (el.matches(excludeSel)) return NodeFilter.FILTER_REJECT;
        el = el.parentElement;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  const re = /\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]/g;
  for (const node of nodes) {
    const text = node.nodeValue;
    re.lastIndex = 0;
    const matches = [];
    let m;
    while ((m = re.exec(text)) !== null) matches.push(m);
    if (!matches.length) continue;
    const frag = document.createDocumentFragment();
    let last = 0;
    for (const m of matches) {
      if (m.index > last)
        frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      frag.appendChild(document.createTextNode("["));
      const innerRe = /(\d+)|([,\u2013-])/g;
      let im;
      let innerLast = 0;
      while ((im = innerRe.exec(m[1])) !== null) {
        if (im.index > innerLast)
          frag.appendChild(document.createTextNode(m[1].slice(innerLast, im.index)));
        if (im[1]) {
          const a = document.createElement("a");
          a.className = "cite";
          a.href = "#ref-" + im[1];
          a.textContent = im[1];
          frag.appendChild(a);
        } else {
          frag.appendChild(document.createTextNode(im[0]));
        }
        innerLast = im.index + im[0].length;
      }
      if (innerLast < m[1].length)
        frag.appendChild(document.createTextNode(m[1].slice(innerLast)));
      frag.appendChild(document.createTextNode("]"));
      last = m.index + m[0].length;
    }
    if (last < text.length)
      frag.appendChild(document.createTextNode(text.slice(last)));
    node.parentNode.replaceChild(frag, node);
  }
}

// Scale equations down to fit the column on large screens (no scrollbar).
function fitEquations() {
  const desktop = window.matchMedia("(min-width: 48rem)").matches;
  document.querySelectorAll(".equation").forEach((el) => {
    el.style.fontSize = "";
    if (!desktop) return;
    const content = el.querySelector(".katex");
    if (!content) return;
    const available = el.clientWidth;
    const wide = content.scrollWidth;
    if (wide > available) {
      const scale = (available / wide) * 0.98;
      el.style.fontSize = Math.floor(100 * scale) + "%";
    }
  });
}

function init() {
  const content = document.querySelector(".content");
  if (content) linkifyCitations(content);
  fitEquations();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(fitEquations);
}
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(fitEquations, 120);
});