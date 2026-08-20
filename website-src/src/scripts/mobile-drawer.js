const drawer = document.getElementById("mobile-drawer");
const backdrop = document.querySelector(".mobile-backdrop");
const toggleBtn = document.getElementById("mobile-menu-btn");
const closeBtn = document.getElementById("mobile-menu-close");

function openDrawer() {
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  backdrop.hidden = false;
  backdrop.classList.add("open");
  toggleBtn.setAttribute("aria-expanded", "true");
  document.body.style.overflow = "hidden";
}

function closeDrawer() {
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  backdrop.hidden = true;
  backdrop.classList.remove("open");
  toggleBtn.setAttribute("aria-expanded", "false");
  document.body.style.overflow = "";
}

toggleBtn.addEventListener("click", () => {
  if (drawer.classList.contains("open")) {
    closeDrawer();
  } else {
    openDrawer();
  }
});

closeBtn.addEventListener("click", closeDrawer);
backdrop.addEventListener("click", closeDrawer);
drawer.addEventListener("click", (e) => {
  if (e.target.closest("a")) closeDrawer();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeDrawer();
});