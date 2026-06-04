function loadPage(page) {
  fetch(page)
    .then(response => response.text())
    .then(data => {
      document.getElementById("main-content").innerHTML = data;
      window.scrollTo(0, 0);
    });
}

// 첫 진입 시 home.html 로드
window.onload = () => {
  loadPage("home.html");
};
