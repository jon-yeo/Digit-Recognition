import React, { useRef, useEffect, useState } from 'react';

const DrawingPad = () => {
    const canvasRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [context, setContext] = useState(null);
    const [drawingData, setDrawingData] = useState([]);
    const [resultNumber, setResultNumber] = useState(null); // State to hold the result number

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        setContext(ctx);

        // Set canvas dimensions
        canvas.width = 280; // 28 cells * 10px each
        canvas.height = 280; // 28 cells * 10px each

        // Initialize drawing data as a 2D array filled with 0
        const initialData = Array.from({ length: 28 }, () => Array(28).fill(-1));
        setDrawingData(initialData);

        // Draw the grid
        drawGrid(ctx);
    }, []);

    const drawGrid = (ctx) => {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // Clear the canvas
        ctx.strokeStyle = 'lightgray';
        ctx.lineWidth = 1;

        for (let i = 0; i <= 28; i++) {
            // Vertical lines
            ctx.beginPath();
            ctx.moveTo(i * 10, 0);
            ctx.lineTo(i * 10, 280);
            ctx.stroke();

            // Horizontal lines
            ctx.beginPath();
            ctx.moveTo(0, i * 10);
            ctx.lineTo(280, i * 10);
            ctx.stroke();
        }
    };

    const startDrawing = (e) => {
        setIsDrawing(true);
        draw(e);
    };

    const endDrawing = () => {
        setIsDrawing(false);
        context.beginPath();
    };

    const draw = (e) => {
        if (!isDrawing) return;

        const rect = canvasRef.current.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / 10); // Map to grid cell
        const y = Math.floor((e.clientY - rect.top) / 10); // Map to grid cell

        if (x < 0 || x >= 28 || y < 0 || y >= 28) return;

        context.fillStyle = 'black';
        context.fillRect(x * 10, y * 10, 10, 10); // Fill the cell

        updateDrawingData(x, y);
    };

    const updateDrawingData = (x, y) => {
        const newDrawingData = drawingData.map(row => row.slice());
        newDrawingData[y][x] = 1; // Mark the cell as drawn
        setDrawingData(newDrawingData);
    };

    const downsampleArray = (largeArray) => {
        const smallSize = 28;
        const downsampledArray = Array.from({ length: smallSize }, () => Array(smallSize).fill(-1));

        // No need to downsample since we are directly working with 28x28
        return largeArray;
    };

    const saveDrawing = async () => {
        // Convert the 2D array to JSON for storage or further processing
        const downsampledData = downsampleArray(drawingData);
        const downsampledArray = JSON.stringify(downsampledData);

        try {
            const response = await fetch('http://localhost:8080/save-drawing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: downsampledArray, // Send the drawing data

            });
            console.log(downsampledArray)
            if (response.ok) {
                const jsonResponse = await response.json();
                console.log('Success:', jsonResponse);
                console.log('Result Number:', jsonResponse.result_number);
                setResultNumber(jsonResponse.result_number);
            } else {
                console.error('Error sending drawing data:', response.statusText);
            }
        } catch (error) {
            console.error('Network error:', error);
        }
    };

    const clearDrawing = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
        drawGrid(ctx); // Redraw the grid

        // Reset the drawing data
        const initialData = Array.from({ length: 28 }, () => Array(28).fill(-1));
        setDrawingData(initialData);
        setResultNumber(null); // Reset the result number
    };

    return (
        <div>
            <canvas
                ref={canvasRef}
                onMouseDown={startDrawing}
                onMouseUp={endDrawing}
                onMouseMove={draw}
                style={{ border: '1px solid black' }}
            />
            <div>
                <button onClick={saveDrawing}>Show Number</button>
                <button onClick={clearDrawing} style={{ marginLeft: '10px' }}>Clear Drawing</button>
                {resultNumber !== null && (
                    <div style={{ marginTop: '10px', fontSize: '18px' }}>
                        Result Number: {resultNumber}
                    </div>
                )}
            </div>

        </div>
    );
};

export default DrawingPad;
