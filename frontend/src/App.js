import React, { useState } from 'react';
import styled from 'styled-components';
import { CSSTransition } from 'react-transition-group';
import './App.css';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background-color: #282c34;
`;

const Input = styled.input`
  margin: 10px;
  padding: 10px;
  border: none;
  border-radius: 5px;
  outline: none;
  transition: background-color 0.3s;
  &:hover {
    background-color: #444;
  }
`;

const ErrorMessage = styled.div`
  color: red;
  margin-top: 20px;
`;

function App() {
  const [inputValues, setInputValues] = useState(Array(5).fill(''));
  const [showInputs, setShowInputs] = useState(false);
  const [error, setError] = useState(null);

  React.useEffect(() => {
    setTimeout(() => {
      setShowInputs(true);
    }, 500);
  }, []);

  const handleInputChange = (index, value) => {
    const newValues = [...inputValues];
    newValues[index] = value;
    setInputValues(newValues);
  };

  const handleKeyPress = async (e) => {
    if (e.key === 'Enter' && inputValues.every(val => val.trim() !== '')) {
      // Call the API (replace with your API endpoint)
      const response = await fetch('http://localhost:8000/api/endpoint', {
        method: 'POST',
        body: JSON.stringify({ data: inputValues }),
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        setError('An error occurred. Please try again.');
      }
    }
  };

  return (
    <Container>
      <h1 style={{ color: '#fff' }}>LoLPredictor</h1>
      {Array.from({ length: 5 }).map((_, index) => (
        <CSSTransition
          in={showInputs}
          timeout={300}
          classNames="fade"
          unmountOnExit
          key={index}
        >
          <Input
            placeholder={`Input ${index + 1}`}
            value={inputValues[index]}
            onChange={(e) => handleInputChange(index, e.target.value)}
            onKeyPress={handleKeyPress}
          />
        </CSSTransition>
      ))}
      {error && <ErrorMessage>{error}</ErrorMessage>}
    </Container>
  );
}

export default App;