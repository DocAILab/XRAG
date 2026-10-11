<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';

const props = defineProps({
  modelValue: String, options: Array, id: String, label: String, placeholder: String,
  optionLabel: { type: Function, default: value => value },
});
const emit = defineEmits(['update:modelValue']);
const root = ref(null);
const open = ref(false);
const active = ref(-1);
const filtering = ref(false);
const candidates = computed(() => {
  const options = props.options || [];
  const query = (props.modelValue || '').toLowerCase();
  return filtering.value && query ? options.filter(value =>
    value.toLowerCase().includes(query) || props.optionLabel(value).toLowerCase().includes(query)) : options;
});
function choose(value) {
  emit('update:modelValue', value);
  open.value = false;
  active.value = -1;
}
function input(event) {
  const text = event.target.value;
  const matchingOption = (props.options || []).find(value => props.optionLabel(value) === text);
  emit('update:modelValue', matchingOption ?? text);
  filtering.value = true;
  open.value = true;
  active.value = -1;
}
function focus() {
  filtering.value = false;
  open.value = true;
}
function keydown(event) {
  if (event.key === 'Escape' || event.key === 'Tab') {
    open.value = false;
    return;
  }
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault();
    if (!open.value) focus();
    const count = candidates.value.length;
    if (!count) return;
    active.value = event.key === 'ArrowDown'
      ? (active.value + 1) % count
      : (active.value < 0 ? count - 1 : (active.value + count - 1) % count);
    requestAnimationFrame(() => root.value?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' }));
  } else if (event.key === 'Enter' && open.value && active.value >= 0) {
    event.preventDefault();
    choose(candidates.value[active.value]);
  }
}
function outside(event) { if (!root.value?.contains(event.target)) open.value = false; }
onMounted(() => document.addEventListener('pointerdown', outside));
onBeforeUnmount(() => document.removeEventListener('pointerdown', outside));
</script>

<template>
  <div ref="root" class="model-combobox">
    <input :id="id" :value="optionLabel(modelValue || '')" type="text" role="combobox" autocomplete="off"
      aria-autocomplete="list" :aria-label="label" :aria-expanded="open && candidates.length > 0"
      :aria-controls="`${id}-list`" :aria-activedescendant="open && active >= 0 ? `${id}-option-${active}` : undefined"
      :placeholder="placeholder" @input="input" @focus="focus" @click="focus" @keydown="keydown" />
    <ul v-if="open && candidates.length" :id="`${id}-list`" class="model-combobox-list" role="listbox" :aria-label="label">
      <li v-for="(value, index) in candidates" :id="`${id}-option-${index}`" :key="value"
        role="option" :aria-selected="index === active" @pointerdown.prevent="choose(value)"
        @pointermove="active = index">{{ optionLabel(value) }}</li>
    </ul>
  </div>
</template>
