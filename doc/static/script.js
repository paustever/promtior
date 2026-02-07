async function ask() {
  const question = document.getElementById("question").value;
  const answerDiv = document.getElementById("answer");
  const loading = document.getElementById("loading");

  answerDiv.innerText = "";
  loading.classList.remove("hidden");

  const response = await fetch("https://TU_API_RAILWAY_URL/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });

  const data = await response.json();
  loading.classList.add("hidden");
  answerDiv.innerText = data.answer;
}
