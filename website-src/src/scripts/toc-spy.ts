const links = Array.from(document.querySelectorAll<HTMLAnchorElement>(".toc a"));
const sections = links
  .map((link) => document.getElementById(link.getAttribute("href")!.slice(1)))
  .filter((el): el is HTMLElement => el !== null);

const setActive = (id: string) => {
  links.forEach((link) => {
    const active = link.getAttribute("href") === `#${id}`;
    link.classList.toggle("active", active);
  });
};

const observer = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        setActive(entry.target.id);
      }
    }
  },
  { rootMargin: "-10% 0px -75% 0px", threshold: 0 }
);

sections.forEach((section) => observer.observe(section));