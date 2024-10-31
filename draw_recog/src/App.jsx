import { useState, useEffect } from 'react'
import './App.css'
import axios from "axios"

import DrawingPad from './widgets/drawing_pad';

function App() {
  const [count, setCount] = useState(0)
  const [number, setNumber] = useState(0)

  const fetchAPI = async () => {
    const response = await axios.get("http://localhost:8080/api/number")
    console.log(response.data.number)
  }

  useEffect(() => {
    fetchAPI();
  }, [])
  return (
    <>
      <h1>Number recognition</h1>
      <div>
        <h3>Draw a number</h3>
        <DrawingPad />
      </div>
    </>
  )
}

export default App
