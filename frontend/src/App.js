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

function App() {
  const [showInputs, setShowInputs] = useState(false);

  // Simulate the fade-in after the component is mounted
  React.useEffect(() => {
    setTimeout(() => {
      setShowInputs(true);
    }, 500);
  }, []);

  return (
    <Container>
      <h1 style={{ color: '#ffffff' }}>lolpredictor</h1>
      <p style={{color : "#ffffff"}}>Hi </p>
      {Array.from({ length: 5 }).map((_, index) => (
        <CSSTransition
          in={showInputs}
          timeout={300}
          classNames="fade"
          unmountOnExit
          key={index}
        >
          <Input
            placeholder={`Summoner ${index + 1}`}
            onChange={(e) => {
              fetch(`http://localhost:8000/api/presentData${index + 1}`, {
                method: 'POST',
                body: JSON.stringify({ data: e.target.value }),
                headers: {
                  'Content-Type': 'application/json',
                },
              });
            }}
          />
        </CSSTransition>
      ))}
    </Container>
  );
}

export default App;