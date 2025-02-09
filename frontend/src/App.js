
import React, { useState, useEffect, useRef } from "react";
import * as d3 from "d3";
import './App.css'; // Import the CSS file
const CustomerSegmentation = () => {
  const [age, setAge] = useState(30);
  const [income, setIncome] = useState(50);
  const [spendingScore, setSpendingScore] = useState(50);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const svgRef = useRef();

  const handleSubmit = async () => {
    const inputData = {
      age: age,
      income: income,
      spending_score: spendingScore,
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(inputData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      setError("Failed to fetch data from the backend");
    }
  };

  // D3.js plot generation
  useEffect(() => {
    if (result) {
      const data = result.encoded_data;  // Assuming the backend returns data for the scatter plot
      const clusters = result.cluster;  // Assuming this contains cluster assignments for each point

      const width = 600;
      const height = 400;
      const margin = 40;

      const svg = d3.select(svgRef.current)
        .attr("width", width)
        .attr("height", height);

      // Set up the scales
      const xScale = d3.scaleLinear()
        .domain([d3.min(data, d => d[0]), d3.max(data, d => d[0])])
        .range([margin, width - margin]);

      const yScale = d3.scaleLinear()
        .domain([d3.min(data, d => d[1]), d3.max(data, d => d[1])])
        .range([height - margin, margin]);

      // Create scatter plot
      svg.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .attr("cx", d => xScale(d[0]))
        .attr("cy", d => yScale(d[1]))
        .attr("r", 5)
        .attr("fill", (d, i) => d3.schemeCategory10[clusters[i] % 10]);

      // Add axes
      svg.append("g")
        .attr("transform", `translate(0, ${height - margin})`)
        .call(d3.axisBottom(xScale));

      svg.append("g")
        .attr("transform", `translate(${margin}, 0)`)
        .call(d3.axisLeft(yScale));
    }
  }, [result]);

  return (
    <div>
      <h1>Customer Segmentation</h1>
      <label>Age:</label>
      <input
        type="number"
        value={age}
        onChange={(e) => setAge(Number(e.target.value))}
      />
      <label>Income:</label>
      <input
        type="number"
        value={income}
        onChange={(e) => setIncome(Number(e.target.value))}
      />
      <label>Spending Score:</label>
      <input
        type="number"
        value={spendingScore}
        onChange={(e) => setSpendingScore(Number(e.target.value))}
      />
      <button onClick={handleSubmit}>Get Segmentation</button>

      {result && (
        <div>
          <h2>Prediction Result</h2>
          <p>Cluster: {result.cluster}</p>
          <p>Encoded Data: {JSON.stringify(result.encoded_data)}</p>
        </div>
      )}

      {error && <p>{error}</p>}

      <svg ref={svgRef}></svg><label>Footer</label>
    </div>
  );
};

export default CustomerSegmentation;
