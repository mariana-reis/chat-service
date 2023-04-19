package entity

import (
	"errors"
	"time"

	"github.com/google/uuid"
	tiktoken_go "github.com/j178/tiktoken-go"
)

type Message struct {
	ID       string
	Role     string
	Content  string
	Tokens   int
	Model    *Model
	CreateAt time.Time
}

func NewMessage(role, content string, model *Model) (*Message, error) {
	totalTokens := tiktoken_go.CountTokens(model.GetModelName(), content)

	msg := &Message{
		ID:       uuid.New().String(),
		Role:     role,
		Content:  content,
		Tokens:   totalTokens,
		Model:    model,
		CreateAt: time.Now(),
	}

	if err := msg.Validate(); err != nil {
		return nil, err
	}

	return msg, nil
}

func (m *Message) Validate() error {
	if m.Role != "user" && m.Role != "system" && m.Role != "assistant" {
		return errors.New("invalid role")
	}

	if m.Content == "" {
		return errors.New("empty content")
	}

	if m.CreateAt.IsZero() {
		return errors.New("invalid create at")
	}

	return nil
}

func (m *Message) GetQtdTokens() int {
	return m.Tokens
}
