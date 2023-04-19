package entity

type Model struct {
	Name     string
	MaxToken int
}

func NewModel(name string, maxToken int) *Model {
	return &Model{
		Name:     name,
		MaxToken: maxToken,
	}
}

func (m *Model) GetMaxToken() int {
	return m.MaxToken
}

func (m *Model) GetModelName() string {
	return m.Name
}
